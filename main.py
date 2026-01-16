#!/usr/bin/env python3
"""
SenseVoice Writer - Push-to-talk voice dictation using SenseVoiceSmall.
Hold the hotkey to record, release to transcribe and type/copy.
Wayland and PipeWire only.
"""

import argparse
import logging
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time

import evdev
from evdev import UInput, ecodes
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

__version__ = "0.1.0"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

POSTPROCESS_PROMPT = """Clean up this speech transcript. Fix grammar, punctuation, and capitalization. Remove filler words (um, uh, like, you know). Output only the cleaned text, nothing else.

Transcript: {transcript}

Cleaned:"""

KEY_MAP = {
    "f1": ecodes.KEY_F1,
    "f2": ecodes.KEY_F2,
    "f3": ecodes.KEY_F3,
    "f4": ecodes.KEY_F4,
    "f5": ecodes.KEY_F5,
    "f6": ecodes.KEY_F6,
    "f7": ecodes.KEY_F7,
    "f8": ecodes.KEY_F8,
    "f9": ecodes.KEY_F9,
    "f10": ecodes.KEY_F10,
    "f11": ecodes.KEY_F11,
    "f12": ecodes.KEY_F12,
    "scroll_lock": ecodes.KEY_SCROLLLOCK,
    "pause": ecodes.KEY_PAUSE,
    "insert": ecodes.KEY_INSERT,
}


def get_hotkey(key_name):
    """Map key name to evdev key code."""
    key_name = key_name.lower()
    if key_name in KEY_MAP:
        return KEY_MAP[key_name]
    if len(key_name) == 1:
        key_attr = f"KEY_{key_name.upper()}"
        if hasattr(ecodes, key_attr):
            return getattr(ecodes, key_attr)
    print(f"Unknown key: {key_name}, defaulting to f12")
    return ecodes.KEY_F12


def get_key_name(keycode):
    """Get human-readable name for a key code."""
    for name, code in KEY_MAP.items():
        if code == keycode:
            return name.upper()
    name = ecodes.KEY.get(keycode, f"KEY_{keycode}")
    if isinstance(name, list):
        name = name[0]
    return name.replace("KEY_", "")


def copy_to_clipboard(text):
    """Copy text to clipboard using wl-copy."""
    process = subprocess.Popen(["wl-copy"], stdin=subprocess.PIPE)
    process.communicate(input=text.encode())


def type_text(text):
    """Type text into active window using wtype."""
    subprocess.run(["wtype", text])


def get_record_command(output_file):
    """Get pw-record command for PipeWire audio capture."""
    return [
        "pw-record",
        "--format",
        "s16",
        "--rate",
        "16000",
        "--channels",
        "1",
        output_file,
    ]


def find_keyboards():
    """Find all keyboard input devices."""
    keyboards = []
    for path in evdev.list_devices():
        try:
            device = evdev.InputDevice(path)
            caps = device.capabilities()
            if ecodes.EV_KEY in caps:
                keys = caps[ecodes.EV_KEY]
                if ecodes.KEY_A in keys or ecodes.KEY_SPACE in keys:
                    keyboards.append(device)
                    logger.debug(f"Found keyboard: {device.path} - {device.name}")
        except (PermissionError, OSError) as e:
            logger.debug(f"Cannot access {path}: {e}")
    return keyboards


def create_uinput(keyboards):
    """Create a virtual keyboard for re-injecting events."""
    all_caps = {}
    for kb in keyboards:
        caps = kb.capabilities()
        for event_type, codes in caps.items():
            if event_type == ecodes.EV_SYN:
                continue
            if event_type not in all_caps:
                all_caps[event_type] = set()
            if isinstance(codes, list):
                for code in codes:
                    if isinstance(code, tuple):
                        all_caps[event_type].add(code[0])
                    else:
                        all_caps[event_type].add(code)
            else:
                all_caps[event_type].add(codes)
    caps_for_uinput = {k: list(v) for k, v in all_caps.items()}
    return UInput(caps_for_uinput, name="SenseVoice Virtual Keyboard")


class Dictation:
    def __init__(self, hotkey, auto_type, grab, notifications, postprocess):
        self.hotkey = hotkey
        self.auto_type = auto_type
        self.grab = grab
        self.notifications = notifications
        self.postprocess = postprocess

        self.recording = False
        self.record_process = None
        self.temp_file = None
        self.model = None
        self.model_loaded = threading.Event()
        self.model_error = None
        self.llm = None
        self.llm_loaded = threading.Event()
        self.llm_error = None
        self.running = True
        self.keyboards = []
        self.selector = None
        self.uinput = None

        print("Loading SenseVoiceSmall model...")
        threading.Thread(target=self._load_model, daemon=True).start()

        if self.postprocess:
            print("Loading Qwen3 model for post-processing...")
            threading.Thread(target=self._load_llm, daemon=True).start()
        else:
            self.llm_loaded.set()

    def _load_model(self):
        try:
            self.model = AutoModel(
                model="FunAudioLLM/SenseVoiceSmall",
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device="cpu",
                hub="hf",
            )
            self.model_loaded.set()
            hotkey_name = get_key_name(self.hotkey)
            print("Model loaded. Ready for dictation!")
            print(f"Hold [{hotkey_name}] to record, release to transcribe.")
            print("Press Ctrl+C to quit.")
        except Exception as e:
            self.model_error = str(e)
            self.model_loaded.set()
            print(f"Failed to load model: {e}")

    def _load_llm(self):
        try:
            model_path = hf_hub_download(
                repo_id="Qwen/Qwen3-0.6B-GGUF",
                filename="Qwen3-0.6B-Q8_0.gguf",
            )
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=4,
                verbose=False,
            )
            self.llm_loaded.set()
            print("Qwen3 model loaded.")
        except Exception as e:
            self.llm_error = str(e)
            self.llm_loaded.set()
            print(f"Failed to load Qwen3 model: {e}")

    def _postprocess_text(self, text):
        """Clean up transcript using LLM."""
        if not self.llm or self.llm_error:
            return text
        try:
            prompt = POSTPROCESS_PROMPT.format(transcript=text)
            result = self.llm(
                prompt,
                max_tokens=len(text) * 2,
                stop=["\n\n", "Transcript:"],
                echo=False,
            )
            cleaned = result["choices"][0]["text"].strip()
            return cleaned if cleaned else text
        except Exception as e:
            logger.debug(f"Post-processing failed: {e}")
            return text

    def notify(self, title, message, icon="dialog-information", timeout=2000):
        """Send a desktop notification."""
        if not self.notifications:
            return
        subprocess.run(
            [
                "notify-send",
                "-a",
                "SenseVoice",
                "-i",
                icon,
                "-t",
                str(timeout),
                "-h",
                "string:x-canonical-private-synchronous:sensevoice",
                title,
                message,
            ],
            capture_output=True,
        )

    def start_recording(self):
        if self.recording or self.model_error:
            return

        self.recording = True
        self.record_start_time = time.perf_counter()
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.temp_file.close()

        record_cmd = get_record_command(self.temp_file.name)
        logger.debug(f"Running: {' '.join(record_cmd)}")
        self.record_process = subprocess.Popen(
            record_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print("Recording...")
        hotkey_name = get_key_name(self.hotkey)
        self.notify(
            "Recording...",
            f"Release {hotkey_name} when done",
            "audio-input-microphone",
            30000,
        )

    def stop_recording(self):
        if not self.recording:
            return

        self.recording = False
        record_duration = time.perf_counter() - self.record_start_time

        if self.record_process:
            self.record_process.terminate()
            self.record_process.wait()
            self.record_process = None

        logger.debug(f"Recording duration: {record_duration:.2f}s")

        print("Transcribing...")
        self.notify(
            "Transcribing...", "Processing your speech", "emblem-synchronizing", 30000
        )

        self.model_loaded.wait()

        if self.model_error:
            print("Cannot transcribe: model failed to load")
            self.notify("Error", "Model failed to load", "dialog-error", 3000)
            return

        try:
            transcribe_start = time.perf_counter()
            res = self.model.generate(
                input=self.temp_file.name,
                cache={},
                language="en",
                use_itn=True,
                beam_size=5,
                merge_vad=True,
            )
            text = rich_transcription_postprocess(res[0]["text"])
            transcribe_duration = time.perf_counter() - transcribe_start
            logger.debug(f"Transcription duration: {transcribe_duration:.2f}s")

            if text:
                if self.postprocess:
                    self.llm_loaded.wait()
                    if not self.llm_error:
                        print("Post-processing...")
                        postprocess_start = time.perf_counter()
                        text = self._postprocess_text(text)
                        postprocess_duration = time.perf_counter() - postprocess_start
                        logger.debug(
                            f"Post-processing duration: {postprocess_duration:.2f}s"
                        )

                copy_to_clipboard(text)
                if self.auto_type:
                    type_text(text)
                print(f"Copied: {text}")
                self.notify(
                    "Copied!",
                    text[:100] + ("..." if len(text) > 100 else ""),
                    "emblem-ok-symbolic",
                    3000,
                )
            else:
                print("No speech detected")
                self.notify(
                    "No speech detected", "Try speaking louder", "dialog-warning", 2000
                )
        except Exception as e:
            print(f"Error: {e}")
            self.notify("Error", str(e)[:50], "dialog-error", 3000)
        finally:
            if self.temp_file and os.path.exists(self.temp_file.name):
                os.unlink(self.temp_file.name)

    def handle_event(self, event):
        """Handle keyboard event."""
        if event.type == ecodes.EV_KEY and event.code == self.hotkey:
            if event.value == 1:  # Key press
                self.start_recording()
            elif event.value == 0:  # Key release
                self.stop_recording()
            return

        if self.grab and self.uinput:
            self.uinput.write_event(event)
            if event.type != ecodes.EV_SYN:
                self.uinput.syn()

    def cleanup(self):
        """Release grabbed devices and close uinput."""
        if self.grab:
            for kb in self.keyboards:
                try:
                    kb.ungrab()
                except OSError:
                    pass
            if self.uinput:
                self.uinput.close()

    def stop(self):
        print("\nExiting...")
        self.running = False
        self.cleanup()
        os.kill(os.getpid(), signal.SIGKILL)

    def run(self):
        self.keyboards = find_keyboards()
        if not self.keyboards:
            print("Error: No keyboards found!")
            print("Make sure you're in the 'input' group: sudo usermod -aG input $USER")
            sys.exit(1)

        if self.grab:
            try:
                self.uinput = create_uinput(self.keyboards)
            except OSError as e:
                print(f"Error creating virtual keyboard: {e}")
                print("Try: sudo modprobe uinput")
                sys.exit(1)

            for kb in self.keyboards:
                try:
                    kb.grab()
                except OSError as e:
                    print(f"Warning: Could not grab {kb.name}: {e}")

            print(
                f"Monitoring {len(self.keyboards)} keyboard(s) with hotkey suppression..."
            )
        else:
            print(f"Monitoring {len(self.keyboards)} keyboard(s)...")

        self.selector = selectors.DefaultSelector()
        for kb in self.keyboards:
            self.selector.register(kb, selectors.EVENT_READ)

        try:
            while self.running:
                for key, mask in self.selector.select(timeout=1):
                    device = key.fileobj
                    try:
                        for event in device.read():
                            self.handle_event(event)
                    except OSError:
                        logger.debug(f"Device disconnected: {device.name}")
                        self.selector.unregister(device)
        finally:
            self.cleanup()


def check_dependencies(auto_type):
    """Check that required system commands are available."""
    missing = []

    if subprocess.run(["which", "pw-record"], capture_output=True).returncode != 0:
        missing.append(("pw-record", "pipewire"))

    if auto_type:
        if subprocess.run(["which", "wtype"], capture_output=True).returncode != 0:
            missing.append(("wtype", "wtype"))

    if subprocess.run(["which", "wl-copy"], capture_output=True).returncode != 0:
        missing.append(("wl-copy", "wl-clipboard"))

    if missing:
        print("Missing dependencies:")
        for cmd, pkg in missing:
            print(f"  {cmd} - install: {pkg}")
        sys.exit(1)

    if not os.environ.get("WAYLAND_DISPLAY"):
        print("Error: WAYLAND_DISPLAY not set. This tool only supports Wayland.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="SenseVoice Writer - Push-to-talk voice dictation using SenseVoiceSmall"
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "-k",
        "--key",
        default="f12",
        help="Hotkey to hold for recording (default: f12)",
    )
    parser.add_argument(
        "--no-type",
        action="store_true",
        help="Disable auto-typing (only copy to clipboard)",
    )
    parser.add_argument(
        "-g",
        "--grab",
        action="store_true",
        help="Grab keyboard exclusively (suppresses hotkey from other apps)",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Disable desktop notifications",
    )
    parser.add_argument(
        "-p",
        "--postprocess",
        action="store_true",
        help="Enable LLM post-processing to clean up transcript (uses Qwen3-0.6B)",
    )
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    print(f"SenseVoice Writer v{__version__}")

    auto_type = not args.no_type
    notifications = not args.no_notify
    hotkey = get_hotkey(args.key)

    logger.debug(f"Hotkey: {args.key}, Auto-type: {auto_type}, Grab: {args.grab}")

    check_dependencies(auto_type)

    dictation = Dictation(
        hotkey=hotkey,
        auto_type=auto_type,
        grab=args.grab,
        notifications=notifications,
        postprocess=args.postprocess,
    )

    def handle_sigint(sig, frame):
        dictation.stop()

    signal.signal(signal.SIGINT, handle_sigint)

    dictation.run()


if __name__ == "__main__":
    main()
