# Jarvis - Local AI Assistant

A Python-based voice-activated personal assistant inspired by JARVIS from Iron Man. This assistant runs entirely on your local machine and can integrate with Ollama for enhanced conversational AI capabilities.

## Features

- **Voice Interaction**: Speak to Jarvis and get voice responses
- **Keyboard Input Mode**: Type commands if you prefer or lack microphone access
- **Local LLM Integration**: Connect to Ollama for advanced conversational AI
- **Memory & Context**: Jarvis remembers your preferences, goals, tasks, and notes
- **Reminders**: Set time-based reminders with natural language
- **Web Integration**: Open websites, search the web, launch applications
- **Planning Assistant**: Get help planning and organizing tasks
- **Calculator**: Perform quick calculations
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Prerequisites

- Python 3.12 or higher (3.11+ should work, but 3.12 recommended)
- (Optional) [Ollama](https://ollama.ai/) for local LLM capabilities
- (Optional) Microphone for voice input
- (Optional) Text-to-speech engine for your OS

## Installation

1. Clone the repository:
```bash
git clone https://github.com/25001502/Jarvis.git
cd Jarvis
```

2. Create a Python virtual environment:
```bash
# Windows
py -3.12 -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3.12 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: PyAudio (required for microphone input) may need system dependencies on Linux:
```bash
# Ubuntu/Debian
sudo apt-get install python3-pyaudio portaudio19-dev

# Fedora
sudo dnf install python3-pyaudio portaudio-devel

# macOS (with Homebrew)
brew install portaudio
```

## Quick Start

### Windows (Voice Mode)
```bash
start_jarvis_voice.bat
```

### Linux/Mac (Voice Mode)
```bash
./start_jarvis_voice.sh
```

### Manual Start (Any Platform)
```bash
# Voice mode with microphone
python jarvis.py

# Keyboard input with voice output
python jarvis.py --keyboard-input

# Text-only mode (no voice)
python jarvis.py --text-only
```

## Usage Examples

Once Jarvis is running, try these commands:

### Personal Information
- "My name is Alex"
- "I prefer concise answers"
- "What is my name?"
- "Show my preferences"

### Memory & Context
- "My goal is to learn Python"
- "I am working on a web scraper"
- "Remember that I have a meeting at 3 PM"
- "Show notes"

### Reminders
- "Remind me to stretch in 30 minutes"
- "Remind me to call mom tomorrow at 2 PM"
- "Show reminders"
- "Complete reminder [ID]"
- "Delete reminder [ID]"

### Web & Applications
- "Open Google"
- "Search for local AI models"
- "Open notepad"
- "Open https://github.com"

### Planning & Productivity
- "Plan build a chatbot"
- "Help me plan learning web development"
- "List files"
- "What time is it?"
- "What is the date?"

### Calculator
- "Calculate 25 * 4 + 10"
- "What is (100 - 25) / 5"

### Multi-Step Commands
- "Execute open google then search artificial intelligence"
- "First open youtube and then search Python tutorials"

### System Information
- "System status"
- "LLM status"
- "Help"

## Configuration

### Command-Line Options

```bash
python jarvis.py [options]

Options:
  --text-only              Disable microphone input
  --keyboard-input         Use keyboard input with voice output
  --list-mics             List available microphones and exit
  --mic-index INDEX       Specify microphone device index
  --self-test             Run environment checks and exit
  --no-llm                Disable local Ollama integration
  --model MODEL           Ollama model name (default: llama3.2:1b)
  --ollama-url URL        Ollama server URL (default: http://127.0.0.1:11434)
  --llm-timeout SECONDS   LLM response timeout (default: 60)
  --llm-max-tokens N      Max tokens for LLM responses (default: 96)
  --memory-file PATH      Path to memory file (default: .jarvis_memory.json)
  --history-turns N       Conversation history length (default: 12)
  --listen-timeout SEC    Speech wait timeout (default: 3.0)
  --phrase-time-limit SEC Max spoken command duration (default: 6.0)
```

### Ollama Integration

To use the conversational AI features:

1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull a model:
```bash
ollama pull llama3.2:1b
```
3. Start Jarvis (it will auto-detect Ollama)

You can use different models:
```bash
python jarvis.py --model llama3.2:3b
python jarvis.py --model mistral
```

## Troubleshooting

### Microphone Issues
```bash
# List available microphones
python jarvis.py --list-mics

# Use specific microphone
python jarvis.py --mic-index 1
```

### Voice Output Issues
- Windows: Uses SAPI5 (built-in)
- macOS: Uses `say` command (built-in)
- Linux: Requires `espeak` or `spd-say`
```bash
# Ubuntu/Debian
sudo apt-get install espeak

# Fedora
sudo dnf install espeak
```

### PyAudio Installation Issues
If PyAudio fails to install, you can use keyboard-input mode:
```bash
python jarvis.py --keyboard-input
```

### Run Self-Test
```bash
python jarvis.py --self-test
```

## Project Structure

```
Jarvis/
├── jarvis.py                   # Main application
├── requirements.txt            # Python dependencies
├── start_jarvis_voice.bat      # Windows launcher
├── start_jarvis_voice.sh       # Linux/Mac launcher
├── .jarvis_memory.json         # Persistent memory (auto-created)
└── README.md                   # This file
```

## Memory Persistence

Jarvis stores your information in `.jarvis_memory.json`:
- User profile (name, preferences)
- Goals and current tasks
- Notes
- Recent commands
- Reminders
- Conversation history

This file is created automatically and persists between sessions.

## Privacy & Security

- All processing happens locally on your machine
- No data is sent to external servers (except optional Ollama on localhost)
- Memory file contains your personal data - keep it secure
- Add `.jarvis_memory.json` to `.gitignore` if committing code

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Inspired by JARVIS from the Marvel Cinematic Universe
- Uses [Ollama](https://ollama.ai/) for local LLM capabilities
- Voice recognition powered by Google Speech Recognition API
- Text-to-speech via pyttsx3 and system TTS engines
