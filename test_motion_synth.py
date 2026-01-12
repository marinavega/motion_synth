"""
Unit tests for Motion Synth components.
Run with: python -m pytest test_motion_synth.py
"""
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from motion_synth import (
    AudioEngine,
    MotionDetector,
    MotionSynth,
    AudioConfig,
    VideoConfig,
    WaveForm
)


class TestAudioConfig:
    """Test AudioConfig dataclass."""
    
    def test_default_values(self):
        config = AudioConfig()
        assert config.sample_rate == 44100
        assert config.base_freq == 110.0
        assert config.max_freq == 1760.0
        assert config.min_volume == 0.05
        assert config.max_volume == 1.0
    
    def test_custom_values(self):
        config = AudioConfig(
            sample_rate=48000,
            base_freq=220.0,
            max_freq=880.0
        )
        assert config.sample_rate == 48000
        assert config.base_freq == 220.0
        assert config.max_freq == 880.0


class TestVideoConfig:
    """Test VideoConfig dataclass."""
    
    def test_default_values(self):
        config = VideoConfig()
        assert config.motion_threshold == 4000
        assert config.smoothing == 0.7
        assert config.blur_kernel == (5, 5)
        assert config.threshold_value == 20
        assert config.dilate_iterations == 3


class TestAudioEngine:
    """Test AudioEngine functionality."""
    
    def test_initialization(self):
        config = AudioConfig()
        engine = AudioEngine(config)
        
        assert engine.config == config
        assert engine.state["frequency"] == 440.0
        assert engine.state["volume"] == 0.2
        assert engine.state["running"] is True
        assert engine.state["waveform"] == WaveForm.SINE
        assert engine.phase == 0.0
    
    def test_set_frequency(self):
        engine = AudioEngine(AudioConfig())
        
        # Normal frequency
        engine.set_frequency(440.0)
        assert engine.state["frequency"] == 440.0
        
        # Clipping - too low
        engine.set_frequency(50.0)
        assert engine.state["frequency"] == 110.0
        
        # Clipping - too high
        engine.set_frequency(5000.0)
        assert engine.state["frequency"] == 1760.0
    
    def test_set_volume(self):
        engine = AudioEngine(AudioConfig())
        
        # Normal volume
        engine.set_volume(0.5)
        assert engine.state["volume"] == 0.5
        
        # Clipping - too low
        engine.set_volume(-0.1)
        assert engine.state["volume"] == 0.05
        
        # Clipping - too high
        engine.set_volume(2.0)
        assert engine.state["volume"] == 1.0
    
    def test_cycle_waveform(self):
        engine = AudioEngine(AudioConfig())
        
        assert engine.state["waveform"] == WaveForm.SINE
        
        waveform = engine.cycle_waveform()
        assert waveform == WaveForm.SQUARE.value
        assert engine.state["waveform"] == WaveForm.SQUARE
        
        engine.cycle_waveform()
        assert engine.state["waveform"] == WaveForm.SAWTOOTH
        
        engine.cycle_waveform()
        assert engine.state["waveform"] == WaveForm.TRIANGLE
        
        engine.cycle_waveform()
        assert engine.state["waveform"] == WaveForm.SINE
    
    def test_toggle_vibrato(self):
        engine = AudioEngine(AudioConfig())
        
        assert engine.state["vibrato_enabled"] is False
        
        result = engine.toggle_vibrato()
        assert result is True
        assert engine.state["vibrato_enabled"] is True
        
        result = engine.toggle_vibrato()
        assert result is False
        assert engine.state["vibrato_enabled"] is False
    
    def test_generate_sine_waveform(self):
        config = AudioConfig(sample_rate=1000)
        engine = AudioEngine(config)
        engine.state["waveform"] = WaveForm.SINE
        
        wave = engine.generate_waveform(440.0, 100)
        
        assert len(wave) == 100
        assert wave.dtype == np.float64
        assert -1 <= wave.min() <= 1
        assert -1 <= wave.max() <= 1
    
    def test_generate_square_waveform(self):
        config = AudioConfig(sample_rate=1000)
        engine = AudioEngine(config)
        engine.state["waveform"] = WaveForm.SQUARE
        
        wave = engine.generate_waveform(440.0, 100)
        
        assert len(wave) == 100
        # Square wave should be mostly -1 or 1
        assert np.all((wave == 1) | (wave == -1) | (np.abs(wave) < 0.1))
    
    def test_generate_waveform_with_vibrato(self):
        config = AudioConfig(sample_rate=1000)
        engine = AudioEngine(config)
        engine.state["vibrato_enabled"] = True
        engine.state["vibrato_rate"] = 5.0
        engine.state["vibrato_depth"] = 0.02
        
        wave = engine.generate_waveform(440.0, 100)
        
        assert len(wave) == 100
        assert -1 <= wave.min() <= 1
        assert -1 <= wave.max() <= 1


class TestMotionDetector:
    """Test MotionDetector functionality."""
    
    @patch('cv2.VideoCapture')
    def test_initialization_success(self, mock_capture):
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_capture.return_value = mock_cap
        
        detector = MotionDetector(VideoConfig(), camera_id=0)
        result = detector.initialize()
        
        assert result is True
        assert detector.width == 640
        assert detector.height == 480
    
    @patch('cv2.VideoCapture')
    def test_initialization_failure(self, mock_capture):
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap
        
        detector = MotionDetector(VideoConfig(), camera_id=0)
        result = detector.initialize()
        
        assert result is False
    
    def test_toggle_debug(self):
        detector = MotionDetector(VideoConfig())
        
        assert detector.show_debug is True
        
        detector.toggle_debug()
        assert detector.show_debug is False
        
        detector.toggle_debug()
        assert detector.show_debug is True


class TestMotionSynth:
    """Test MotionSynth main application."""
    
    def test_initialization(self):
        audio_config = AudioConfig()
        video_config = VideoConfig()
        
        app = MotionSynth(audio_config, video_config, camera_id=0)
        
        assert app.audio_config == audio_config
        assert app.running is False
        assert app.paused is False
    
    def test_map_position_to_audio(self):
        app = MotionSynth(AudioConfig(), VideoConfig())
        app.motion_detector.width = 640
        app.motion_detector.height = 480
        
        # Center position
        freq, vol = app.map_position_to_audio(320, 240)
        assert 110.0 < freq < 1760.0
        assert 0.05 < vol < 1.0
        
        # Top-left corner (high pitch, high volume)
        freq, vol = app.map_position_to_audio(0, 0)
        assert freq > 1500.0  # Near max
        assert vol > 0.8      # Near max
        
        # Bottom-right corner (low pitch, low volume)
        freq, vol = app.map_position_to_audio(640, 480)
        assert freq < 300.0   # Near min
        assert vol < 0.2      # Near min
    
    def test_handle_keypress_waveform(self):
        app = MotionSynth(AudioConfig(), VideoConfig())
        
        initial_waveform = app.audio_engine.state["waveform"]
        result = app.handle_keypress(ord('w'))
        
        assert result is True
        assert app.audio_engine.state["waveform"] != initial_waveform
    
    def test_handle_keypress_vibrato(self):
        app = MotionSynth(AudioConfig(), VideoConfig())
        
        initial_vibrato = app.audio_engine.state["vibrato_enabled"]
        result = app.handle_keypress(ord('v'))
        
        assert result is True
        assert app.audio_engine.state["vibrato_enabled"] != initial_vibrato
    
    def test_handle_keypress_pause(self):
        app = MotionSynth(AudioConfig(), VideoConfig())
        
        assert app.paused is False
        app.handle_keypress(ord(' '))
        assert app.paused is True
        app.handle_keypress(ord(' '))
        assert app.paused is False
    
    def test_handle_keypress_debug(self):
        app = MotionSynth(AudioConfig(), VideoConfig())
        
        initial_debug = app.motion_detector.show_debug
        result = app.handle_keypress(ord('d'))
        
        assert result is True
        assert app.motion_detector.show_debug != initial_debug
    
    def test_handle_keypress_exit(self):
        app = MotionSynth(AudioConfig(), VideoConfig())
        
        result = app.handle_keypress(27)  # ESC key
        assert result is False


class TestIntegration:
    """Integration tests."""
    
    def test_audio_video_coordination(self):
        """Test that audio engine and motion detector can work together."""
        audio_config = AudioConfig()
        video_config = VideoConfig()
        app = MotionSynth(audio_config, video_config)
        
        # Simulate motion detection
        app.motion_detector.width = 640
        app.motion_detector.height = 480
        
        # Map position to audio
        freq, vol = app.map_position_to_audio(320, 240)
        
        # Update audio engine
        app.audio_engine.set_frequency(freq)
        app.audio_engine.set_volume(vol)
        
        assert app.audio_engine.state["frequency"] == freq
        assert app.audio_engine.state["volume"] == vol
    
    def test_full_waveform_cycle(self):
        """Test cycling through all waveforms."""
        engine = AudioEngine(AudioConfig())
        waveforms = [WaveForm.SINE, WaveForm.SQUARE, WaveForm.SAWTOOTH, WaveForm.TRIANGLE]
        
        for expected_waveform in waveforms:
            assert engine.state["waveform"] == expected_waveform
            engine.cycle_waveform()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
