import cv2
import mediapipe as mp

# 입력 영상 경로
video_path = "시선비디오1.mov"

# 출력 영상 경로
output_path = "example_skeleton.mp4"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ 비디오를 열 수 없습니다.")
    raise SystemExit

# 비디오 정보
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 저장용 VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


# 관절 좌표 저장 리스트
skeleton_data = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    keypoints = []

    if result.pose_landmarks:
        # 관절 그리기
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_style.get_default_pose_landmarks_style()
        )

        # 33개 관절 좌표 저장
        for lm in result.pose_landmarks.landmark:
            keypoints.append((lm.x, lm.y, lm.z, lm.visibility))
    else:
        # 관절이 없으면 None 저장
        keypoints = None

    skeleton_data.append(keypoints)

    # 영상 저장
    out.write(frame)

    cv2.imshow("Skeleton Tracking", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Skeleton 영상 저장 완료:", output_path)


# 좌표 JSON 저장
import json
with open("skeleton_coords.json", "w") as f:
    json.dump(skeleton_data, f, indent=4)

print("✅ Skeleton 좌표 저장 완료: skeleton_coords.json")
