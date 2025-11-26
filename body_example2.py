import cv2
import mediapipe as mp
import json

video_path = "시선비디오1.mov"
output_video = "example_upperbody_holistic2.mp4"
output_json = "upperbody_skeleton.json"

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ 비디오를 열 수 없습니다.")
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# 상반신 + 손 좌표 저장
# frame별로 { "pose": [...], "left_hand": [...], "right_hand": [...] } 구조
all_frames = []

# Pose 상반신 인덱스 (어깨~골반, 팔)
UPPER_BODY_POSE_IDX = [
    0,      # nose (참고용)
    11, 12, # shoulders
    13, 14, # elbows
    15, 16, # wrists
    23, 24  # hips (상반신 기준 축으로 사용)
]

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)

        # 시각화: 포즈 + 양손만 그림
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_style.get_default_pose_landmarks_style()
            )
        if result.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                result.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )
        if result.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                result.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        frame_info = {
            "pose": None,
            "left_hand": None,
            "right_hand": None
        }

        # 상반신 pose 좌표만 저장
        if result.pose_landmarks:
            pose_points = []
            for idx in UPPER_BODY_POSE_IDX:
                lm = result.pose_landmarks.landmark[idx]
                pose_points.append({
                    "idx": idx,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "vis": lm.visibility
                })
            frame_info["pose"] = pose_points

        # 왼손 21점
        if result.left_hand_landmarks:
            left_hand_points = []
            for i, lm in enumerate(result.left_hand_landmarks.landmark):
                left_hand_points.append({
                    "idx": i,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z
                })
            frame_info["left_hand"] = left_hand_points

        # 오른손 21점
        if result.right_hand_landmarks:
            right_hand_points = []
            for i, lm in enumerate(result.right_hand_landmarks.landmark):
                right_hand_points.append({
                    "idx": i,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z
                })
            frame_info["right_hand"] = right_hand_points

        all_frames.append(frame_info)

        out.write(frame)
        cv2.imshow("Upper-body + Hands (Holistic)", frame)
        if cv2.waitKey(1) == 27:
            break

cap.release()
out.release()
cv2.destroyAllWindows()

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(all_frames, f, indent=2, ensure_ascii=False)

print("✅ 상반신+손 스켈레톤 영상 저장:", output_video)
print("✅ 상반신+손 좌표 JSON 저장:", output_json)
