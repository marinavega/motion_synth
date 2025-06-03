import cv2
import numpy as np
import sounddevice as sd
import time

SAMPLE_RATE = 44100
BASE_FREQ = 220
MAX_FREQ = 880
MIN_VOLUME = 0.1
MAX_VOLUME = 1.0

def play_tone(freq, duration, volume):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.sin(freq * t * 2 * np.pi)
    audio = wave * volume
    sd.play(audio, samplerate=SAMPLE_RATE)
    sd.wait()

def map_position_to_sound(x, y, frame_width, frame_height, prev_freq):
    pitch_position = 1.0 - (x / frame_width)
    freq = BASE_FREQ + pitch_position * (MAX_FREQ - BASE_FREQ)
    freq = 0.85 * prev_freq + 0.15 * freq

    volume_position = 1.0 - (y / frame_height)
    volume = MIN_VOLUME + volume_position * (MAX_VOLUME - MIN_VOLUME)

    return freq, volume

def process_frame(frame1, frame2, frame_width, frame_height, prev_freq):
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 1000:
            x, y, w, h = cv2.boundingRect(largest)
            center_x = x + w // 2
            center_y = y + h // 2

            freq, volume = map_position_to_sound(center_x, center_y, frame_width, frame_height, prev_freq)
            print(f"X: {center_x}, Y: {center_y} → {freq:.1f} Hz @ vol {volume:.2f}")
            play_tone(freq=freq, duration=0.15, volume=volume)

            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame1, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(frame1, f"{freq:.1f} Hz / {volume:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            return freq

    return prev_freq

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open cam.")
        return

    time.sleep(1)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    frame_height, frame_width = frame1.shape[:2]

    prev_freq = 440
    print("Move hand: left/right = pitch, up/down = volume. Press ESC to exit.")

    try:
        while cap.isOpened():
            prev_freq = process_frame(frame1, frame2, frame_width, frame_height, prev_freq)
            cv2.imshow("Motion Controller", frame1)

            frame1 = frame2
            ret, frame2 = cap.read()

            if cv2.waitKey(10) == 27:
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
