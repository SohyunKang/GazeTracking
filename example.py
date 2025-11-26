"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import json

gaze = GazeTracking()
webcam = cv2.VideoCapture(2)
if not webcam.isOpened():

    print('fail')

# 비디오 경로 설정
video_path = 'example.mp4'

# 출력 프레임 저장 폴더
# output_dir = './frames'
# os.makedirs(output_dir, exist_ok=True)

# 비디오 열기
cap = cv2.VideoCapture(video_path)

# 프레임 수와 FPS 정보 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"🎞 FPS: {fps}, Total Frames: {frame_count}")

frame_idx = 0

# while True:
#     _, frame = cap.read()
left = []
right = []

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    if not left_pupil:
        left.append(list('nan'))
    else:
        left.append(list(float(x) for x in left_pupil))

    if not right_pupil:
        right.append(list('nan'))
    else:
        right.append(list(float(x) for x in right_pupil))

    cv2.imshow("Demo", frame)

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump({'left': left, 'right': right}, f, ensure_ascii=False, indent=4)

    if cv2.waitKey(1) == 27:
        break



webcam.release()
cv2.destroyAllWindows()
