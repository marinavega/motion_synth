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
    "running": True
}

prev_x, prev_y = None, None

def audio_loop():
    phase = 0.0
    def callback(outdata, frames, time_info, status):
        nonlocal phase
        freq = audio_state["frequency"]
        vol = audio_state["volume"]
        increment = 2 * np.pi * freq / SAMPLE_RATE
        wave = np.sin(phase + increment * np.arange(frames))
        phase += increment * frames
        outdata[:] = (wave * vol).reshape(-1, 1)
    with sd.OutputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE):
        while audio_state["running"]:
            time.sleep(0.01)

def map_position_to_audio(x, y, w, h):
    pitch_ratio = 1.0 - x / w
    freq = BASE_FREQ + pitch_ratio * (MAX_FREQ - BASE_FREQ)
    v_ratio = 1.0 - y / h
    volume = MIN_VOLUME + (v_ratio ** 2) * (MAX_VOLUME - MIN_VOLUME)
    return freq, volume

def main():
    global prev_x, prev_y
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    h, w = frame1.shape[:2]
    thread = threading.Thread(target=audio_loop)
    thread.start()
    try:
        print("Theremin ready. Move your hand — press ESC to exit.")
        while cap.isOpened():
            diff = cv2.absdiff(frame1, frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=3)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area > MOTION_THRESHOLD:
                    x, y, bw, bh = cv2.boundingRect(largest)
                    cx = x + bw // 2
                    cy = y + bh // 2
                    if prev_x is None:
                        prev_x, prev_y = cx, cy
                    else:
                        cx = int(SMOOTHING * prev_x + (1 - SMOOTHING) * cx)
                        cy = int(SMOOTHING * prev_y + (1 - SMOOTHING) * cy)
                        prev_x, prev_y = cx, cy
                    freq, vol = map_position_to_audio(cx, cy, w, h)
                    audio_state["frequency"] = freq
                    audio_state["volume"] = vol
                    print(f"[🎛️] Motion | Freq: {freq:.1f} Hz | Volume: {vol:.2f}")
                    cv2.rectangle(frame1, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv2.circle(frame1, (cx, cy), 5, (0, 0, 255), -1)
                else:
                    print("[⚠️] Motion too small")
            else:
                print("[🟡] No motion detected")
            cv2.imshow("Theremin", frame1)
            frame1 = frame2
            ret, frame2 = cap.read()
            if cv2.waitKey(10) == 27:
                break
    finally:
        print("Shutting down...")
        audio_state["running"] = False
        thread.join()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
