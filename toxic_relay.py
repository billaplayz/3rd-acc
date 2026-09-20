import asyncio
import math
import os
from array import array
from collections import deque
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from telethon import TelegramClient, events
from telethon.sessions import StringSession

from pytgcalls import PyTgCalls
from pytgcalls import filters as tg_filters

from pytgcalls.types import (
    Device,
    Direction,
    ExternalMedia,
    MediaStream,
    RecordStream,
    StreamFrames,
)

from pytgcalls.types.raw import AudioParameters


# ============================================================
# CONFIG
# ============================================================

API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
OWNER_ID = int(os.getenv("OWNER_ID", "0"))

PHONE = os.getenv("PHONE_NUMBER", "").strip()
PREFIX = os.getenv("PREFIX", "$")

SESSION_STRING = os.getenv("SESSION_STRING", "").strip()
SESSION_PATH = os.getenv(
    "SESSION_PATH",
    "./data/toxic_relay"
)

if not API_ID or not API_HASH or not BOT_TOKEN or not OWNER_ID:
    raise SystemExit(
        "Missing API_ID, API_HASH, BOT_TOKEN or OWNER_ID "
        "in Railway Variables/.env"
    )

Path(SESSION_PATH).parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GLOBALS
# ============================================================

user = None
bot = None
call = None

relay = {}
relay_filter_state = {}

phone_queue = None
code_queue = None
password_queue = None


# ============================================================
# AUDIO
# ============================================================

SAMPLE_RATE = 48000
CHANNELS = 2


# ============================================================
# FX SETTINGS
# ============================================================

fx = {
    "volume": 100.0,
    "gain": 0.0,
    "loudness": 100.0,

    "bass": 100.0,
    "treble": 100.0,

    "presence": 100.0,
    "clarity": 100.0,

    "echo": "off",
    "reverb": "off",

    "compressor": "on",
    "gate": "off",
    "agc": "on",

    "widen": 100.0,
    "delay": 0.0,

    "ghostmute": "off",
}


# ============================================================
# HELPERS
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def new_dsp_state():
    return {
        "low": [0.0, 0.0],
        "high_lp": [0.0, 0.0],
        "presence_lp": [0.0, 0.0],

        "echo": deque(
            maxlen=int(
                SAMPLE_RATE *
                0.7 *
                CHANNELS
            )
        ),

        "delay": deque(
            maxlen=int(
                SAMPLE_RATE *
                0.25 *
                CHANNELS
            )
        ),

        "rms": 0.08,
    }


def state_for(source):
    return relay_filter_state.setdefault(
        source,
        new_dsp_state()
    )


# ============================================================
# FX STATUS
# ============================================================

def settings_text():

    return (
        "🔊 <b>TOXIC RELAY FX</b>\n\n"

        f"Volume: <b>{fx['volume']:g}%</b>\n"
        f"Gain: <b>{fx['gain']:g} dB</b>\n"
        f"Loudness: <b>{fx['loudness']:g}%</b>\n"

        f"Bass: <b>{fx['bass']:g}%</b>\n"
        f"Treble: <b>{fx['treble']:g}%</b>\n"

        f"Presence: <b>{fx['presence']:g}%</b>\n"
        f"Clarity: <b>{fx['clarity']:g}%</b>\n"

        f"Compressor: <b>{fx['compressor']}</b>\n"
        f"AGC: <b>{fx['agc']}</b>\n"
        f"Gate: <b>{fx['gate']}</b>\n"

        f"Echo: <b>{fx['echo']}</b>\n"
        f"Reverb: <b>{fx['reverb']}</b>\n"

        f"Widen: <b>{fx['widen']:g}%</b>\n"
        f"Delay: <b>{fx['delay']:g} ms</b>\n"

        f"Ghost mute: <b>{fx['ghostmute']}</b>"
    )


# ============================================================
# DSP PROCESSOR
# ============================================================

def process_pcm(pcm: bytes, source: int) -> bytes:

    if not pcm:
        return pcm

    usable = len(pcm) - (
        len(pcm) % 2
    )

    samples = array("h")

    samples.frombytes(
        pcm[:usable]
    )

    if not samples:
        return pcm

    st = state_for(source)

    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    sum_sq = 0.0

    for sample in samples:

        x = sample / 32768.0

        sum_sq += x * x

    frame_rms = math.sqrt(
        sum_sq /
        max(1, len(samples))
    )

    st["rms"] = (
        st["rms"] * 0.94
        +
        frame_rms * 0.06
    )

    # --------------------------------------------------------
    # NO ARTIFICIAL COMMAND LIMITS
    # --------------------------------------------------------

    volume = (
        max(0.0, fx["volume"])
        / 100.0
    )

    gain_db = fx["gain"]

    loudness = (
        max(0.0, fx["loudness"])
        / 100.0
    )

    bass_strength = (
        max(0.0, fx["bass"])
        - 100.0
    ) / 100.0

    treble_strength = (
        max(0.0, fx["treble"])
        - 100.0
    ) / 100.0

    presence_strength = (
        max(0.0, fx["presence"])
        - 100.0
    ) / 100.0

    clarity_strength = (
        max(0.0, fx["clarity"])
        - 100.0
    ) / 100.0

    # --------------------------------------------------------
    # AGC
    # --------------------------------------------------------

    agc = 1.0

    if fx["agc"] == "on":

        target = 0.11

        agc = clamp(
            target /
            max(
                st["rms"],
                0.012
            ),
            0.55,
            3.2
        )

    # --------------------------------------------------------
    # GAIN
    # --------------------------------------------------------

    try:

        linear_gain = (
            10.0 **
            (gain_db / 20.0)
        )

    except OverflowError:

        linear_gain = (
            float("inf")
            if gain_db > 0
            else 0.0
        )

    linear = (
        linear_gain
        *
        volume
        *
        loudness
        *
        agc
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    low_a = (
        1.0 -
        math.exp(
            -2.0 *
            math.pi *
            180.0 /
            SAMPLE_RATE
        )
    )

    high_a = (
        1.0 -
        math.exp(
            -2.0 *
            math.pi *
            3400.0 /
            SAMPLE_RATE
        )
    )

    presence_a = (
        1.0 -
        math.exp(
            -2.0 *
            math.pi *
            1700.0 /
            SAMPLE_RATE
        )
    )

    # --------------------------------------------------------
    # DELAY
    # --------------------------------------------------------

    delay_ms = max(
        0.0,
        fx["delay"]
    )

    delay_samples = int(
        SAMPLE_RATE *
        delay_ms /
        1000.0
    ) * CHANNELS

    echo_delay = (
        int(
            SAMPLE_RATE *
            0.32
        )
        * CHANNELS
    )

    reverb_delay = (
        int(
            SAMPLE_RATE *
            0.08
        )
        * CHANNELS
    )

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    processed = [
        0.0
    ] * len(samples)

    for i, raw in enumerate(samples):

        ch = i % CHANNELS

        x = raw / 32768.0

        # LOW
        low = (
            st["low"][ch]
            +
            low_a *
            (
                x -
                st["low"][ch]
            )
        )

        st["low"][ch] = low

        # HIGH
        high_lp = (
            st["high_lp"][ch]
            +
            high_a *
            (
                x -
                st["high_lp"][ch]
            )
        )

        st["high_lp"][ch] = high_lp

        high = (
            x -
            high_lp
        )

        # PRESENCE
        pres_lp = (
            st["presence_lp"][ch]
            +
            presence_a *
            (
                x -
                st["presence_lp"][ch]
            )
        )

        st["presence_lp"][ch] = pres_lp

        presence = (
            pres_lp -
            low
        )

        # ----------------------------------------------------
        # EQ
        # ----------------------------------------------------

        y = (
            x
            +
            low *
            bass_strength
            +
            high *
            treble_strength
        )

        y += (
            presence
            *
            presence_strength
            *
            0.55
        )

        y += (
            high
            *
            clarity_strength
            *
            0.35
        )

        # ----------------------------------------------------
        # MASTER GAIN
        # ----------------------------------------------------

        y *= linear

        # ----------------------------------------------------
        # NOISE GATE
        # ----------------------------------------------------

        if (
            fx["gate"] == "on"
            and
            abs(y) < 0.012
            and
            st["rms"] < 0.035
        ):

            y *= 0.18

        # ----------------------------------------------------
        # COMPRESSOR
        # ----------------------------------------------------

        if fx["compressor"] == "on":

            amplitude = abs(y)

            threshold = 0.32

            if amplitude > threshold:

                compressed = (
                    threshold
                    +
                    (
                        amplitude -
                        threshold
                    ) / 3.5
                )

                y = math.copysign(
                    compressed,
                    y
                )

        # ----------------------------------------------------
        # ECHO
        # ----------------------------------------------------

        if (
            fx["echo"] == "on"
            and
            len(st["echo"])
            >= echo_delay
        ):

            y += (
                st["echo"][-echo_delay]
                * 0.22
            )

        st["echo"].append(y)

        # ----------------------------------------------------
        # DELAY
        # ----------------------------------------------------

        if (
            delay_samples
            and
            len(st["delay"])
            >= delay_samples
        ):

            y += (
                st["delay"][
                    -delay_samples
                ]
                * 0.12
            )

        st["delay"].append(y)

        # ----------------------------------------------------
        # REVERB
        # ----------------------------------------------------

        if (
            fx["reverb"] == "on"
            and
            len(st["echo"])
            >= reverb_delay
        ):

            y += (
                st["echo"][
                    -reverb_delay
                ]
                * 0.10
            )

        processed[i] = y

    # --------------------------------------------------------
    # STEREO WIDEN
    # --------------------------------------------------------

    widen = (
        max(
            0.0,
            fx["widen"]
        )
        / 100.0
    )

    if (
        CHANNELS == 2
        and
        widen != 1.0
    ):

        for i in range(
            0,
            len(processed) - 1,
            2
        ):

            left = processed[i]

            right = processed[
                i + 1
            ]

            mid = (
                left +
                right
            ) * 0.5

            side = (
                (
                    left -
                    right
                )
                * 0.5
                * widen
            )

            processed[i] = (
                mid + side
            )

            processed[
                i + 1
            ] = (
                mid - side
            )

    # --------------------------------------------------------
    # PCM OUTPUT
    # --------------------------------------------------------

    out = array("h")

    for y in processed:

        # Final PCM safety.
        # This is NOT an FX command limit.

        try:

            y = math.tanh(
                y * 1.12
            ) * 0.985

        except (OverflowError, ValueError):

            y = (
                1.0
                if y > 0
                else -1.0
            )

        y = max(
            -1.0,
            min(
                1.0,
                y
            )
        )

        out.append(
            int(
                y * 32767
            )
        )

    return out.tobytes()


# ============================================================
# INCOMING VC FRAME HANDLER
# ============================================================

async def frame_handler(_, update: StreamFrames):

    source = update.chat_id

    cfg = relay.get(source)

    if not cfg:
        return

    if not update.frames:
        return

    target = cfg["target"]

    sample_lists = []

    max_samples = 0

    for frame in update.frames:

        raw = frame.frame

        usable = len(raw) - (
            len(raw) % 2
        )

        samples = array("h")

        samples.frombytes(
            raw[:usable]
        )

        if samples:

            sample_lists.append(
                samples
            )

            max_samples = max(
                max_samples,
                len(samples)
            )

    if not max_samples:
        return

    # --------------------------------------------------------
    # MIX
    # --------------------------------------------------------

    mixed = array(
        "h",
        [0] * max_samples
    )

    count = len(
        sample_lists
    )

    for samples in sample_lists:

        for i, value in enumerate(
            samples
        ):

            mixed[i] += (
                value // count
            )

    mixed = array(
        "h",
        (
            clamp(
                x,
                -32768,
                32767
            )
            for x in mixed
        )
    )

    try:

        audio = process_pcm(
            mixed.tobytes(),
            source
        )

        await call.send_frame(
            target,
            Device.MICROPHONE,
            audio
        )

    except Exception as exc:

        print(
            "[RELAY] send_frame:",
            type(exc).__name__,
            str(exc),
            flush=True
        )


# ============================================================
# START RELAY
# ============================================================

async def relay_on(
    source: int,
    target: int
):

    if source == target:

        return (
            "❌ Source and target VC "
            "must be different."
        )

    if source in relay:

        return (
            "⚠️ Relay is already ON "
            "in this source VC."
        )

    if call is None:

        return (
            "❌ VC engine is not initialized."
        )

    try:

        if not user.is_connected():

            return (
                "❌ Telegram user client "
                "is disconnected. "
                "Restart the Railway service."
            )

        if not await user.is_user_authorized():

            return (
                "❌ Telegram user session "
                "is not authorized."
            )

    except Exception as exc:

        return (
            "❌ User client check failed: "
            f"{type(exc).__name__}: {exc}"
        )

    params = AudioParameters(
        bitrate=SAMPLE_RATE,
        channels=CHANNELS
    )

    try:

        call.enable_action = (
            fx["ghostmute"] != "on"
        )

        await call.record(
            source,
            RecordStream(
                audio=True,
                audio_parameters=params,
                camera=False,
                screen=False,
            )
        )

        await call.play(
            target,
            MediaStream(
                ExternalMedia.AUDIO,
                params
            )
        )

    except Exception as exc:

        return (
            "❌ Relay start failed: "
            f"{type(exc).__name__}: {exc}"
        )

    relay_filter_state[
        source
    ] = new_dsp_state()

    relay[source] = {
        "target": target
    }

    return (
        "🎙️ <b>RELAY ON</b>\n"
        f"Source: <code>{source}</code>\n"
        f"Target: <code>{target}</code>\n"
        "FX: <b>enhanced DSP</b>\n"
        f"Ghost mute: "
        f"<b>{fx['ghostmute']}</b>"
    )


# ============================================================
# STOP RELAY
# ============================================================

async def relay_off(source: int):

    cfg = relay.pop(
        source,
        None
    )

    relay_filter_state.pop(
        source,
        None
    )

    if not cfg:

        return (
            "ℹ️ Relay is already OFF."
        )

    try:

        await call.leave_call(
            source
        )

    except Exception:
        pass

    try:

        await call.leave_call(
            cfg["target"]
        )

    except Exception:
        pass

    return (
        "⛔ <b>RELAY OFF</b>"
    )


# ============================================================
# TELEGRAM LOGIN CALLBACKS
# ============================================================

async def phone_callback():

    if PHONE:
        return PHONE

    await bot.send_message(
        OWNER_ID,
        f"📱 Send "
        f"{PREFIX}phone +91XXXXXXXXXX"
    )

    return await phone_queue.get()


async def code_callback():

    await bot.send_message(
        OWNER_ID,
        f"🔐 Telegram OTP required.\n"
        f"Send {PREFIX}otp 12345"
    )

    return await code_queue.get()


async def password_callback():

    await bot.send_message(
        OWNER_ID,
        f"🔑 2FA password required.\n"
        f"Send {PREFIX}2fa YOUR_PASSWORD"
    )

    return await password_queue.get()


# ============================================================
# USER AUTH
# ============================================================

async def auth_user():

    print(
        "🔐 Starting Telegram user client...",
        flush=True
    )

    try:

        await user.start(
            phone=phone_callback,
            code_callback=code_callback,
            password=password_callback,
        )

    except Exception as exc:

        print(
            "❌ Telegram login failed:",
            type(exc).__name__,
            str(exc),
            flush=True
        )

        raise

    if not user.is_connected():

        raise RuntimeError(
            "Telegram user client is not "
            "connected after start()."
        )

    if not await user.is_user_authorized():

        raise RuntimeError(
            "Telegram user client is "
            "not authorized."
        )

    me = await user.get_me()

    print(
        f"✅ Telegram user connected: "
        f"{me.id}",
        flush=True
    )


# ============================================================
# COMMAND CONTROLLER
# ============================================================

async def controller(event):

    if event.sender_id != OWNER_ID:
        return

    text = (
        event.raw_text or ""
    ).strip()

    if not text.startswith(
        PREFIX
    ):
        return

    parts = text[
        len(PREFIX):
    ].split()

    if not parts:
        return

    cmd = parts[0].lower()

    args = parts[1:]

    # --------------------------------------------------------
    # LOGIN
    # --------------------------------------------------------

    if cmd == "phone":

        if args:

            await phone_queue.put(
                args[0]
            )

            await event.reply(
                "✅ Phone received. "
                "Requesting Telegram code…"
            )

        return

    if cmd == "otp":

        if args:

            await code_queue.put(
                args[0]
            )

            await event.reply(
                "✅ OTP received."
            )

        return

    if cmd == "2fa":

        if args:

            await password_queue.put(
                " ".join(args)
            )

            await event.reply(
                "✅ 2FA received."
            )

        return

    # --------------------------------------------------------
    # MENU
    # --------------------------------------------------------

    if cmd in {
        "start",
        "panel",
        "admin",
        "help"
    }:

        await event.reply(
            menu()
        )

        return

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if cmd == "status":

        active = ", ".join(
            f"{source} → "
            f"{cfg['target']}"
            for source, cfg
            in relay.items()
        )

        if not active:
            active = (
                "No active relay"
            )

        try:

            connected = (
                user.is_connected()
            )

            authorized = (
                await user.is_user_authorized()
            )

        except Exception:

            connected = False
            authorized = False

        await event.reply(
            "🔥 <b>TOXIC RELAY</b>\n\n"
            "Telegram: "
            f"<b>{'CONNECTED' if connected else 'OFFLINE'}</b>\n"
            "Authorized: "
            f"<b>{'YES' if authorized else 'NO'}</b>\n"
            f"Relay: {active}"
        )

        return

    # --------------------------------------------------------
    # NUMERIC FX
    # --------------------------------------------------------

    numeric_commands = {
        "volume",
        "gain",
        "loudness",
        "bass",
        "treble",
        "presence",
        "clarity",
        "widen",
        "delay",
    }

    if cmd in numeric_commands:

        if not args:

            await event.reply(
                f"Usage: "
                f"{PREFIX}{cmd} <value>"
            )

            return

        try:

            value = float(
                args[0]
            )

        except ValueError:

            await event.reply(
                "❌ Value must be "
                "a number."
            )

            return

        # NO ARTIFICIAL LIMITS
        fx[cmd] = value

        await event.reply(
            f"✅ {cmd.upper()} = "
            f"{fx[cmd]:g}"
        )

        return

    # --------------------------------------------------------
    # TOGGLES
    # --------------------------------------------------------

    toggle_commands = {
        "echo",
        "reverb",
        "compressor",
        "gate",
        "agc",
        "ghostmute",
    }

    if cmd in toggle_commands:

        if (
            args
            and
            args[0].lower()
            in {"on", "off"}
        ):

            fx[cmd] = (
                args[0].lower()
            )

            if (
                cmd == "ghostmute"
                and call is not None
            ):

                call.enable_action = (
                    fx["ghostmute"] != "on"
                )

            await event.reply(
                f"✅ {cmd.upper()} "
                f"{fx[cmd]}"
            )

        else:

            await event.reply(
                f"Usage: "
                f"{PREFIX}{cmd} on/off"
            )

        return

    # --------------------------------------------------------
    # RELAY
    # --------------------------------------------------------

    if cmd == "relay":

        if len(args) != 1:

            await event.reply(
                "Usage in source group:\n"
                f"{PREFIX}relay "
                "<target_chat_id>"
            )

            return

        try:

            target = int(
                args[0]
            )

        except ValueError:

            await event.reply(
                "❌ Target chat ID "
                "must be numeric."
            )

            return

        result = await relay_on(
            event.chat_id,
            target
        )

        await event.reply(
            result
        )

        return

    # --------------------------------------------------------
    # LEAVE
    # --------------------------------------------------------

    if cmd in {
        "leave",
        "stop"
    }:

        await event.reply(
            await relay_off(
                event.chat_id
            )
        )

        return

    # --------------------------------------------------------
    # LEAVE ALL
    # --------------------------------------------------------

    if cmd == "leaveall":

        if not relay:

            await event.reply(
                "ℹ️ No active relays."
            )

            return

        results = []

        for source in list(
            relay
        ):

            results.append(
                await relay_off(
                    source
                )
            )

        await event.reply(
            "\n".join(results)
        )

        return

    # --------------------------------------------------------
    # FX STATUS
    # --------------------------------------------------------

    if cmd == "fx":

        await event.reply(
            settings_text()
        )

        return

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    await event.reply(
        menu()
    )


# ============================================================
# MENU
# ============================================================

def menu():

    return (
        "╔══════════════════════════════╗\n"
        "║   🔥 TOXIC RELAY • PYTHON   ║\n"
        "╚══════════════════════════════╝\n\n"

        "🎙 <b>VOICE CHAT RELAY</b>\n\n"

        f"{PREFIX}relay &lt;target_chat_id&gt;\n"
        f"{PREFIX}leave\n"
        f"{PREFIX}leaveall\n"
        f"{PREFIX}status\n\n"

        "🎛 <b>LOUDNESS / VOICE FX</b>\n\n"

        f"{PREFIX}volume &lt;value&gt;\n"
        f"{PREFIX}gain &lt;value dB&gt;\n"
        f"{PREFIX}loudness &lt;value&gt;\n"
        f"{PREFIX}bass &lt;value&gt;\n"
        f"{PREFIX}treble &lt;value&gt;\n"
        f"{PREFIX}presence &lt;value&gt;\n"
        f"{PREFIX}clarity &lt;value&gt;\n"
        f"{PREFIX}widen &lt;value&gt;\n"
        f"{PREFIX}delay &lt;value ms&gt;\n\n"

        f"{PREFIX}compressor on/off\n"
        f"{PREFIX}agc on/off\n"
        f"{PREFIX}gate on/off\n"
        f"{PREFIX}echo on/off\n"
        f"{PREFIX}reverb on/off\n"
        f"{PREFIX}ghostmute on/off\n"
        f"{PREFIX}fx\n\n"

        "🔐 <b>FIRST LOGIN</b>\n"

        f"{PREFIX}phone +91XXXXXXXXXX\n"
        f"{PREFIX}otp 12345\n"
        f"{PREFIX}2fa YOUR_PASSWORD\n\n"

        f"Prefix: <b>{PREFIX}</b>"
    )


# ============================================================
# MAIN
# ============================================================

async def main():

    global user
    global bot
    global call

    global phone_queue
    global code_queue
    global password_queue

    print(
        "🔥 TOXIC RELAY • "
        "RAILWAY DSP BUILD",
        flush=True
    )

    # --------------------------------------------------------
    # USER CLIENT
    # --------------------------------------------------------

    if SESSION_STRING:

        print(
            "🔐 Using SESSION_STRING",
            flush=True
        )

        user = TelegramClient(
            StringSession(
                SESSION_STRING
            ),
            API_ID,
            API_HASH
        )

    else:

        print(
            f"🔐 Using session file: "
            f"{SESSION_PATH}",
            flush=True
        )

        user = TelegramClient(
            SESSION_PATH,
            API_ID,
            API_HASH
        )

    # --------------------------------------------------------
    # BOT
    # --------------------------------------------------------

    bot = TelegramClient(
        None,
        API_ID,
        API_HASH
    )

    # --------------------------------------------------------
    # PYTGCALLS
    # --------------------------------------------------------

    call = PyTgCalls(
        user
    )

    # --------------------------------------------------------
    # QUEUES
    # --------------------------------------------------------

    phone_queue = asyncio.Queue()
    code_queue = asyncio.Queue()
    password_queue = asyncio.Queue()

    # --------------------------------------------------------
    # BOT HANDLER
    # --------------------------------------------------------

    bot.add_event_handler(
        controller,
        events.NewMessage
    )

    # --------------------------------------------------------
    # START BOT
    # --------------------------------------------------------

    await bot.start(
        bot_token=BOT_TOKEN
    )

    print(
        "🟢 Controller bot connected.",
        flush=True
    )

    # --------------------------------------------------------
    # START USER
    # --------------------------------------------------------

    await auth_user()

    if not user.is_connected():

        raise RuntimeError(
            "Telegram user client "
            "is not connected."
        )

    if not await user.is_user_authorized():

        raise RuntimeError(
            "Telegram user client "
            "is not authorized."
        )

    # --------------------------------------------------------
    # START VOICE ENGINE
    # --------------------------------------------------------

    await call.start()

    # --------------------------------------------------------
    # INCOMING VC AUDIO
    # --------------------------------------------------------

    call.on_update(
        tg_filters.stream_frame(
            Direction.INCOMING,
            Device.MICROPHONE,
        )
    )(frame_handler)

    print(
        "🟢 Controller + VC engine online",
        flush=True
    )

    print(
        f"🎙️ Ready for "
        f"{PREFIX}relay",
        flush=True
    )

    # --------------------------------------------------------
    # KEEP ALIVE
    # --------------------------------------------------------

    await asyncio.gather(
        bot.run_until_disconnected(),
        user.run_until_disconnected(),
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "🛑 Stopped.",
            flush=True
        )

    except Exception as exc:

        print(
            "💥 FATAL ERROR:",
            type(exc).__name__,
            str(exc),
            flush=True
        )

        raise