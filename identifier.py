import cv2
import math
import numpy as np
# import torch
from data_loader import masked_face_dataset, ImageTransform
import mediapipe as mp


DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

# mediapipe configs
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('image', img)
    cv2.waitKey(0) # 사용자가 아무 키를 누를때까지 이미지를 계속 띄워줌


# Read images with OpenCV.
# images = {name: cv2.imread(name) for name in uploaded.keys()}
# Preview the images.

# 이미지로 테스트
"""
myDataset = masked_face_dataset(dir_path="dataset/mask_dataset", phase='show',
                        transform=ImageTransform(1024, (0.5, 0.5, 0.5), (0.2, 0.2, 0.2)))
X, y = {}, {}
for i in range(3):
    x_label, x_img, y_label, y_img = myDataset[0]
    x_img = torch.permute(x_img, (1, 2, 0))
    x_img = np.array(x_img.tolist(), dtype=np.float32)
    X[x_label] = x_img
    y[y_label] = y_img

print(X)

for name, image in X.items():
    print(name)
    resize_and_show(image)
"""

# 웹캠으로 테스트
video = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:

    while (True):
        # frame마다 캡쳐하기
        ret, frame = video.read()

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #results = face_mesh.process(cv2.cvtColor(frame))

        # Draw face landmarks of each face.

        if not results.multi_face_landmarks:
            continue
        annotated_image = frame.copy()
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())

        # 화면에 띄우기
        cv2.imshow('frame', annotated_image)

        # 'q'버튼 누르면 종료하기
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    video.release()
    cv2.destroyAllWindows()
