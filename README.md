# Motion Synth

Use your webcam to control pitch and volume with motion.  
Move your hand left/right to change the **pitch**, and up/down to change the **volume** in real time.

---

## 🚀 Features

- Real-time motion detection via webcam
- Tone generation using `sounddevice`
- Map X axis to pitch (220–880 Hz), Y axis to volume (0.1–1.0)
- No dependencies on audio files

---

## 🧰 Requirements

- Python 3.7+
- `opencv-python` (or `opencv-python-headless` on ARM Macs)
- `numpy`
- `sounddevice`

Install all with:

```bash
pip install -r requirements.txt
