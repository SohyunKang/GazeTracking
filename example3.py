import cv2
from gaze_tracking import GazeTracking

# 🔹 입력 비디오 경로
video_path = "시선비디오1.mov"   # <- 여기에 네 비디오 경로

# 🔹 출력(저장) 비디오 경로
output_path = "example_annotated.mp4"

gaze = GazeTracking()
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ 비디오를 열 수 없습니다.")
    raise SystemExit

# 원본 비디오 정보 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 🔹 VideoWriter 설정 (mp4 저장)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("🎬 비디오 끝")
        break

    # GazeTracking 분석
    gaze.refresh(frame)
    frame = gaze.annotated_frame()

    # 상태 텍스트
    text = ""
    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(
        frame, text, (90, 60),
        cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2
    )

    # 동공 좌표
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    cv2.putText(frame, f"Left pupil:  {left_pupil}", (90, 130),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, f"Right pupil: {right_pupil}", (90, 165),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # 🔹 프레임 저장 (annotated 영상)
    out.write(frame)

    # 화면에도 보여주고 싶으면 유지
    cv2.imshow("Gaze Tracking (Video)", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ 저장 완료:", output_path)
