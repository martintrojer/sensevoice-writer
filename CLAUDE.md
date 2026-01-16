# CLAUDE.md

## Project Overview

SenseVoice Writer - A push-to-talk voice dictation tool using SenseVoiceSmall. Wayland and PipeWire only.

## Commands

```bash
# Install dependencies
uv sync

# Run the app
uv run python main.py

# Run with options
uv run python main.py -k pause --no-type -g

# Format code
uv run ruff format .

# Check code
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .
```

## Code Quality

Each commit must be clean:
- `ruff format .` - no formatting changes
- `ruff check .` - no errors or warnings

Run both before committing.

## System Dependencies

- `pw-record` (pipewire)
- `wtype` (for auto-typing)
- `wl-copy` (wl-clipboard)
- User must be in `input` group: `sudo usermod -aG input $USER`
