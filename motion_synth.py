import cv2
import numpy as np
import sounddevice as sd
import threading
import time

SAMPLE_RATE = 44100
BASE_FREQ = 110
MAX_FREQ = 1760
MIN_VOLUME = 0.05
MAX_VOLUME = 1.0
MOTION_THRESHOLD = 4000
SMOOTHING = 0.7

audio_state = {
    "frequency": 440.0,
    "volume": 0.2,
"""
Motion Synth - Webcam-controlled synthesizer
Uses hand motion to control pitch, volume, and effects in real-time.
"""
import cv2
import numpy as np
import sounddevice as sd
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import argparse


class WaveForm(Enum):
    """Available waveform types."""
    SINE = "sine"
    SQUARE = "square"
    SAWTOOTH = "sawtooth"
    TRIANGLE = "triangle"


@dataclass
class AudioConfig:
    """Configuration for audio parameters."""
    sample_rate: int = 44100
    base_freq: float = 110.0
    max_freq: float = 1760.0
    min_volume: float = 0.05
    max_volume: float = 1.0


@dataclass
class VideoConfig:
    """Configuration for video processing."""
    motion_threshold: int = 4000
    smoothing: float = 0.7
    blur_kernel: Tuple[int, int] = (5, 5)
    threshold_value: int = 20
    dilate_iterations: int = 3


class AudioEngine:
    """Handles audio synthesis and playback."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.state = {
            "frequency": 440.0,
            "volume": 0.2,
            "running": True,
            "waveform": WaveForm.SINE,
            "vibrato_enabled": False,
            "vibrato_rate": 5.0,
            "vibrato_depth": 0.02
        }
        self.phase = 0.0
        self.vibrato_phase = 0.0
        self.thread: Optional[threading.Thread] = None
        
    def generate_waveform(self, freq: float, frames: int) -> np.ndarray:
        """Generate waveform based on current settings."""
        increment = 2 * np.pi * freq / self.config.sample_rate
        
        # Apply vibrato if enabled
        if self.state["vibrato_enabled"]:
            vibrato_increment = 2 * np.pi * self.state["vibrato_rate"] / self.config.sample_rate
            vibrato = np.sin(self.vibrato_phase + vibrato_increment * np.arange(frames))
            freq_modulation = freq * (1 + self.state["vibrato_depth"] * vibrato)
            increment = 2 * np.pi * freq_modulation / self.config.sample_rate
            self.vibrato_phase += vibrato_increment * frames
        
        t = self.phase + increment * np.arange(frames)
        
        # Generate waveform
        waveform_type = self.state["waveform"]
        if waveform_type == WaveForm.SINE:
            wave = np.sin(t)
        elif waveform_type == WaveForm.SQUARE:
            wave = np.sign(np.sin(t))
        elif waveform_type == WaveForm.SAWTOOTH:
            wave = 2 * (t / (2 * np.pi) - np.floor(t / (2 * np.pi) + 0.5))
        elif waveform_type == WaveForm.TRIANGLE:
            wave = 2 * np.abs(2 * (t / (2 * np.pi) - np.floor(t / (2 * np.pi) + 0.5))) - 1
        else:
            wave = np.sin(t)
        
        self.phase += increment * frames
        # Keep phase bounded
        if self.phase > 2 * np.pi * 1000:
            self.phase %= 2 * np.pi
        
        return wave
    
    def audio_callback(self, outdata, frames, time_info, status):
        """Callback for audio stream."""
        if status:
            print(f"Audio status: {status}")
        
        freq = self.state["frequency"]
        vol = self.state["volume"]
        
        wave = self.generate_waveform(freq, frames)
        outdata[:] = (wave * vol).reshape(-1, 1)
    
    def start(self):
        """Start the audio engine."""
        self.thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.thread.start()
    
    def _audio_loop(self):
        """Main audio processing loop."""
        with sd.OutputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.config.sample_rate
        ):
            while self.state["running"]:
                time.sleep(0.01)
    
    def stop(self):
        """Stop the audio engine."""
        self.state["running"] = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def set_frequency(self, freq: float):
        """Set the output frequency."""
        self.state["frequency"] = np.clip(
            freq,
            self.config.base_freq,
            self.config.max_freq
        )
    
    def set_volume(self, volume: float):
        """Set the output volume."""
        self.state["volume"] = np.clip(
            volume,
            self.config.min_volume,
            self.config.max_volume
        )
    
    def cycle_waveform(self):
        """Cycle to the next waveform type."""
        waveforms = list(WaveForm)
        current_idx = waveforms.index(self.state["waveform"])
        next_idx = (current_idx + 1) % len(waveforms)
        self.state["waveform"] = waveforms[next_idx]
        return waveforms[next_idx].value
    
    def toggle_vibrato(self):
        """Toggle vibrato effect on/off."""
        self.state["vibrato_enabled"] = not self.state["vibrato_enabled"]
        return self.state["vibrato_enabled"]


class MotionDetector:
    """Handles video capture and motion detection."""
    
    def __init__(self, config: VideoConfig, camera_id: int = 0):
        self.config = config
        self.camera_id = camera_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.prev_x: Optional[int] = None
        self.prev_y: Optional[int] = None
        self.frame1: Optional[np.ndarray] = None
        self.frame2: Optional[np.ndarray] = None
        self.width: int = 0
        self.height: int = 0
        self.show_debug = True
        
    def initialize(self) -> bool:
        """Initialize the video capture."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            return False
        
        # Allow camera to warm up
        time.sleep(1)
        
        # Capture initial frames
        ret, self.frame1 = self.cap.read()
        ret, self.frame2 = self.cap.read()
        
        if not ret or self.frame1 is None:
            return False
        
        self.height, self.width = self.frame1.shape[:2]
        return True
    
    def detect_motion(self) -> Optional[Tuple[int, int]]:
        """
        Detect motion and return the center point of the largest moving object.
        Returns (x, y) coordinates or None if no significant motion detected.
        """
        if self.frame1 is None or self.frame2 is None:
            return None
        
        # Calculate difference between frames
        diff = cv2.absdiff(self.frame1, self.frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, self.config.blur_kernel, 0)
        _, thresh = cv2.threshold(
            blur,
            self.config.threshold_value,
            255,
            cv2.THRESH_BINARY
        )
        dilated = cv2.dilate(thresh, None, iterations=self.config.dilate_iterations)
        
        # Find contours
        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        if area < self.config.motion_threshold:
            return None
        
        # Get bounding box and center
        x, y, w, h = cv2.boundingRect(largest)
        cx = x + w // 2
        cy = y + h // 2
        
        # Apply smoothing
        if self.prev_x is not None:
            cx = int(self.config.smoothing * self.prev_x + (1 - self.config.smoothing) * cx)
            cy = int(self.config.smoothing * self.prev_y + (1 - self.config.smoothing) * cy)
        
        self.prev_x, self.prev_y = cx, cy
        
        # Draw debug visualization
        if self.show_debug:
            cv2.rectangle(self.frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(self.frame1, (cx, cy), 5, (0, 0, 255), -1)
        
        return (cx, cy)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the current frame for display."""
        return self.frame1
    
    def advance_frame(self) -> bool:
        """Advance to the next frame."""
        if self.cap is None or not self.cap.isOpened():
            return False
        
        self.frame1 = self.frame2
        ret, self.frame2 = self.cap.read()
        return ret
    
    def release(self):
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def toggle_debug(self):
        """Toggle debug visualization."""
        self.show_debug = not self.show_debug


class MotionSynth:
    """Main application coordinating audio and video."""
    
    def __init__(
        self,
        audio_config: AudioConfig,
        video_config: VideoConfig,
        camera_id: int = 0
    ):
        self.audio_engine = AudioEngine(audio_config)
        self.motion_detector = MotionDetector(video_config, camera_id)
        self.audio_config = audio_config
        self.running = False
        self.paused = False
        
    def map_position_to_audio(self, x: int, y: int) -> Tuple[float, float]:
        """
        Map screen position to frequency and volume.
        X-axis: pitch (left=high, right=low)
        Y-axis: volume (top=high, bottom=low)
        """
        w = self.motion_detector.width
        h = self.motion_detector.height
        
        # Map X to frequency (inverted: left = high)
        pitch_ratio = 1.0 - (x / w)
        freq = (
            self.audio_config.base_freq +
            pitch_ratio * (self.audio_config.max_freq - self.audio_config.base_freq)
        )
        
        # Map Y to volume (inverted: top = high)
        volume_ratio = 1.0 - (y / h)
        # Use quadratic curve for more natural volume control
        volume = (
            self.audio_config.min_volume +
            (volume_ratio ** 2) * (self.audio_config.max_volume - self.audio_config.min_volume)
        )
        
        return freq, volume
    
    def draw_ui(self, frame: np.ndarray):
        """Draw UI overlay on the frame."""
        h, w = frame.shape[:2]
        
        # Draw control info
        info = [
            f"Waveform: {self.audio_engine.state['waveform'].value}",
            f"Freq: {self.audio_engine.state['frequency']:.1f} Hz",
            f"Volume: {self.audio_engine.state['volume']:.2f}",
            f"Vibrato: {'ON' if self.audio_engine.state['vibrato_enabled'] else 'OFF'}",
            f"{'PAUSED' if self.paused else ''}",
            "",
            "Controls:",
            "W - Cycle waveform",
            "V - Toggle vibrato",
            "D - Toggle debug view",
            "SPACE - Pause/Resume",
            "ESC - Quit"
        ]
        
        y_offset = 30
        for line in info:
            if line:
                cv2.putText(
                    frame,
                    line,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    line,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1
                )
            y_offset += 25
        
        # Draw frequency/volume grid
        if self.motion_detector.show_debug:
            # Vertical lines for pitch
            for i in range(1, 4):
                x = int(w * i / 4)
                cv2.line(frame, (x, 0), (x, h), (100, 100, 100), 1)
            
            # Horizontal lines for volume
            for i in range(1, 4):
                y = int(h * i / 4)
                cv2.line(frame, (0, y), (w, y), (100, 100, 100), 1)
    
    def handle_keypress(self, key: int) -> bool:
        """
        Handle keyboard input.
        Returns False if the application should exit.
        """
        if key == 27:  # ESC
            return False
        elif key == ord('w') or key == ord('W'):
            waveform = self.audio_engine.cycle_waveform()
            print(f"Waveform: {waveform}")
        elif key == ord('v') or key == ord('V'):
            vibrato = self.audio_engine.toggle_vibrato()
            print(f"Vibrato: {'ON' if vibrato else 'OFF'}")
        elif key == ord('d') or key == ord('D'):
            self.motion_detector.toggle_debug()
            print(f"Debug view: {'ON' if self.motion_detector.show_debug else 'OFF'}")
        elif key == ord(' '):
            self.paused = not self.paused
            print(f"{'Paused' if self.paused else 'Resumed'}")
        
        return True
    
    def run(self):
        """Main application loop."""
        # Initialize components
        if not self.motion_detector.initialize():
            print("Error: Could not open camera")
            return
        
        print("Initializing Motion Synth...")
        self.audio_engine.start()
        self.running = True
        
        print("\n" + "="*50)
        print("Motion Synth Ready!")
        print("="*50)
        print("Move your hand to control the synth:")
        print("  • Left/Right → Pitch")
        print("  • Up/Down → Volume")
        print("\nPress 'W' to cycle waveforms")
        print("Press 'V' to toggle vibrato")
        print("Press 'D' to toggle debug view")
        print("Press SPACE to pause/resume")
        print("Press ESC to exit")
        print("="*50 + "\n")
        
        try:
            while self.running and self.motion_detector.cap.isOpened():
                # Detect motion
                position = self.motion_detector.detect_motion()
                
                # Update audio based on motion
                if position is not None and not self.paused:
                    x, y = position
                    freq, vol = self.map_position_to_audio(x, y)
                    self.audio_engine.set_frequency(freq)
                    self.audio_engine.set_volume(vol)
                
                # Display frame with UI
                frame = self.motion_detector.get_frame()
                if frame is not None:
                    self.draw_ui(frame)
                    cv2.imshow("Motion Synth", frame)
                
                # Advance to next frame
                if not self.motion_detector.advance_frame():
                    break
                
                # Handle keyboard input
                key = cv2.waitKey(10) & 0xFF
                if key != 255:
                    if not self.handle_keypress(key):
                        break
        
        finally:
            print("\nShutting down Motion Synth...")
            self.running = False
            self.audio_engine.stop()
            self.motion_detector.release()
            print("Goodbye!")


def main():
    """Entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Motion Synth - Webcam-controlled synthesizer"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--base-freq",
        type=float,
        default=110.0,
        help="Base frequency in Hz (default: 110.0)"
    )
    parser.add_argument(
        "--max-freq",
        type=float,
        default=1760.0,
        help="Maximum frequency in Hz (default: 1760.0)"
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.7,
        help="Motion smoothing factor 0-1 (default: 0.7)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=4000,
        help="Motion detection threshold (default: 4000)"
    )
    
    args = parser.parse_args()
    
    # Create configurations
    audio_config = AudioConfig(
        base_freq=args.base_freq,
        max_freq=args.max_freq
    )
    
    video_config = VideoConfig(
        smoothing=args.smoothing,
        motion_threshold=args.threshold
    )
    
    # Create and run application
    app = MotionSynth(audio_config, video_config, args.camera)
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()