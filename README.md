# Motion Synth 🎹✋

A real-time webcam-controlled synthesizer that transforms hand motion into music. Move your hand to control pitch, volume, and effects without touching any instrument!

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## Features

### Core Functionality
- **Real-time Motion Detection**: Track hand movements via webcam with smooth interpolation
- **Gesture-to-Sound Mapping**: 
  - X-axis (left/right) → Pitch control (110-1760 Hz)
  - Y-axis (up/down) → Volume control (0.05-1.0)
- **Multiple Waveforms**: Sine, Square, Sawtooth, and Triangle waves
- **Vibrato Effect**: Toggle vibrato for expressive modulation
- **Visual Feedback**: On-screen display with frequency grid and controls

### Code Quality Improvements
- **Object-Oriented Architecture**: Clean separation of concerns with dedicated classes
- **Type Hints**: Full type annotations for better IDE support and code clarity
- **Configuration Management**: Dataclass-based configuration for easy customization
- **Error Handling**: Robust error handling and graceful shutdown
- **Command-Line Interface**: Configurable parameters via argparse
- **Thread-Safe Audio**: Proper audio engine with callback-based synthesis
- **Documentation**: Comprehensive docstrings and inline comments

---

## Controls

| Key | Action |
|-----|--------|
| **Mouse Movement** | Control pitch (X) and volume (Y) |
| **W** | Cycle through waveforms (Sine → Square → Sawtooth → Triangle) |
| **V** | Toggle vibrato effect on/off |
| **D** | Toggle debug visualization |
| **SPACE** | Pause/Resume audio |
| **ESC** | Exit application |

---

## Requirements

- **Python 3.7+**
- **opencv-python** or **opencv-python-headless** (for ARM Macs)
- **numpy**
- **sounddevice**

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/motion-synth.git
cd motion-synth

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Quick Start

### Basic Usage

```bash
python3 motion_synth.py
```

### Advanced Usage

```bash
# Use a different camera
python3 motion_synth.py --camera 1

# Customize frequency range
python3 motion_synth.py --base-freq 220 --max-freq 880

# Adjust motion smoothing
python3 motion_synth.py --smoothing 0.5

# Change motion detection sensitivity
python3 motion_synth.py --threshold 2000
```

### Command-Line Options

```
--camera INT          Camera device ID (default: 0)
--base-freq FLOAT     Base frequency in Hz (default: 110.0)
--max-freq FLOAT      Maximum frequency in Hz (default: 1760.0)
--smoothing FLOAT     Motion smoothing factor 0-1 (default: 0.7)
--threshold INT       Motion detection threshold (default: 4000)
```

---

## 🏗️ Architecture

### Project Structure

```
motion-synth/
├── motion_synth.py       # Main application
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── LICENSE              # MIT License
└── .gitignore          # Git ignore rules
```

### Class Diagram

```
┌─────────────────┐
│  MotionSynth    │  ← Main orchestrator
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼────┐ ┌─▼──────────┐
│ Audio  │ │ Motion     │
│ Engine │ │ Detector   │
└────────┘ └────────────┘
```

### Key Components

#### `AudioEngine`
- Manages audio synthesis and playback
- Supports multiple waveforms
- Implements vibrato effect
- Thread-safe audio callback system

#### `MotionDetector`
- Handles webcam capture
- Processes frame differences for motion detection
- Applies smoothing for stable tracking
- Provides visual feedback

#### `MotionSynth`
- Coordinates audio and video components
- Maps motion to audio parameters
- Handles user input and UI rendering

---

## Waveform Types

| Waveform | Character | Best For |
|----------|-----------|----------|
| **Sine** | Pure, smooth | Flute-like tones, meditation |
| **Square** | Hollow, buzzy | Retro game sounds, aggressive tones |
| **Sawtooth** | Bright, harsh | Strings, brass-like sounds |
| **Triangle** | Mellow, flute-like | Softer melodies, woodwinds |

---

## Configuration

### Audio Configuration

```python
AudioConfig(
    sample_rate=44100,      # Audio sample rate
    base_freq=110.0,        # Lowest frequency (A2)
    max_freq=1760.0,        # Highest frequency (A6)
    min_volume=0.05,        # Minimum volume
    max_volume=1.0          # Maximum volume
)
```

### Video Configuration

```python
VideoConfig(
    motion_threshold=4000,   # Minimum motion area to detect
    smoothing=0.7,           # Smoothing factor (0=none, 1=max)
    blur_kernel=(5, 5),      # Gaussian blur size
    threshold_value=20,      # Binary threshold
    dilate_iterations=3      # Morphological dilation
)
```

---

## 🚀 Use Cases

- **Performance Art**: Interactive installations and live performances
- **Music Education**: Teaching pitch and volume concepts visually
- **Accessibility**: Alternative instrument for people with limited mobility
- **Fun Experimentation**: Creative sound exploration
- **Therapy**: Music therapy and rehabilitation exercises

---

## Troubleshooting

### Camera Not Found
```bash
# List available cameras
python3 -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"

# Try different camera ID
python3 motion_synth.py --camera 1
```

### Audio Issues
```bash
# List audio devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"

# Install PortAudio on macOS
brew install portaudio
```

### High CPU Usage
- Reduce camera resolution in `MotionDetector.initialize()`
- Increase smoothing factor: `--smoothing 0.9`
- Lower frame processing rate by modifying `cv2.waitKey()`

---

## 🔮 Future Enhancements

- [ ] MIDI output support
- [ ] Multi-hand tracking
- [ ] Recording and playback
- [ ] Effect presets and saving
- [ ] Web-based interface
- [ ] Mobile app version
- [ ] Gesture recognition for effects
- [ ] Scale/key constraints
- [ ] Reverb and delay effects
- [ ] Multi-voice polyphony

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run with debug output
python3 motion_synth.py --threshold 1000
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by the theremin, one of the first electronic instruments
- Built with OpenCV for computer vision
- Audio synthesis powered by sounddevice and NumPy

---

## Contact

Questions? Suggestions? Feel free to open an issue or reach out via email to marinavega@protonmail.com