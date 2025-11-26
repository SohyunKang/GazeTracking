"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import json
import time

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print('fail')

# ----------------------
# 🎥 Annotated frame 저장을 위한 설정
# ----------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
save_path = "annotated_output.mp4"

fps = 7.0
width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
print(f"Saving annotated video to: {save_path}")

# 10초 타이머
start_time = time.time()

left = []
right = []

while True:
    if time.time() - start_time > 10:
        print("⏹ 10 seconds passed. Recording stopped.")
        break

    ret, frame = webcam.read()
    if not ret:
        break

    # gaze tracking
    gaze.refresh(frame)
    annotated = gaze.annotated_frame()

    text = ""
    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(annotated, text, (90, 60),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    cv2.putText(annotated, "Left pupil:  " + str(left_pupil),
                (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                (147, 58, 31), 1)
    cv2.putText(annotated, "Right pupil: " + str(right_pupil),
                (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                (147, 58, 31), 1)

    # ----------------------
    # ⭐ 주석 포함된 frame 저장 (가장 중요!)
    # ----------------------
    out.write(annotated)

    # pupil 데이터 기록
    if not left_pupil:
        left.append([float('nan'), float('nan')])
    else:
        left.append([float(x) for x in left_pupil])

    if not right_pupil:
        right.append([float('nan'), float('nan')])
    else:
        right.append([float(x) for x in right_pupil])

    cv2.imshow("Demo", annotated)

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump({"left": left, "right": right}, f,
                  ensure_ascii=False, indent=4)

    if cv2.waitKey(1) == 27:
        break

webcam.release()
out.release()
cv2.destroyAllWindows()
