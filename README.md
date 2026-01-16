# SenseVoice Writer

Push-to-talk voice dictation using [SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall). Hold a hotkey to record, release to transcribe and type.

**Wayland and PipeWire only.**

## Installation

### System Dependencies

```bash
# Fedora
sudo dnf install pipewire wtype wl-clipboard

# Arch
sudo pacman -S pipewire wtype wl-clipboard
```

Add yourself to the `input` group for keyboard access:

```bash
sudo usermod -aG input $USER
# Log out and back in
```

### Python Dependencies

```bash
uv sync
```

## Usage

```bash
# Basic usage (F12 hotkey)
uv run python main.py

# Custom hotkey
uv run python main.py -k pause

# Clipboard only (no auto-typing)
uv run python main.py --no-type

# Grab keyboard exclusively (suppress hotkey from other apps)
uv run python main.py -g

# Enable LLM post-processing to clean up transcript
uv run python main.py -p

# Disable notifications
uv run python main.py --no-notify

# Debug mode
uv run python main.py -d
```

## Options

| Option | Description |
|--------|-------------|
| `-k, --key KEY` | Hotkey to hold for recording (default: f12) |
| `--no-type` | Disable auto-typing, only copy to clipboard |
| `-g, --grab` | Grab keyboard exclusively |
| `-p, --postprocess` | Enable LLM post-processing (Qwen3-0.6B GGUF) |
| `--no-notify` | Disable desktop notifications |
| `-d, --debug` | Enable debug logging |
| `-v, --version` | Show version |

## Supported Hotkeys

`f1`-`f12`, `scroll_lock`, `pause`, `insert`, or any single letter/number key.

## How It Works

1. Hold the hotkey to start recording via PipeWire
2. Release to stop recording and transcribe with SenseVoiceSmall
3. (Optional) LLM cleans up the transcript (fixes grammar, removes filler words)
4. Text is copied to clipboard and typed into the active window

## Post-Processing

The `-p` flag enables LLM post-processing using Qwen3-0.6B (GGUF format via llama-cpp-python). This cleans up the transcript by:
- Fixing grammatical errors and typos
- Standardizing punctuation and capitalization
- Removing filler words (um, ah, like)
- Removing technical tags like [laughter]

The model is downloaded automatically on first use from HuggingFace.
