import cv2
import math
import numpy as np
import torch
from data_loader import masked_face_dataset, ImageTransform
import mediapipe as mp


DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
IMG_SIZE = (200, 200)
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
NOSE = [6, 122, 188, 114, 129, 98, 97, 2, 326, 327, 358, 343, 412, 351]
LIP = [0, 37, 39, 61, 84, 17, 314, 291, 267]

# mediapipe configs
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def landmarksDetection(img, face_landmarks, draw=False):
    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [[int(point.x * img_width), int(point.y * img_height)] for point in
                  face_landmarks.landmark]
    if draw:
        [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]
        print(1)

    # returning the list of tuples for each landmarks
    return mesh_coord

def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('image', img)
    cv2.waitKey(0) # 사용자가 아무 키를 누를때까지 이미지를 계속 띄워줌


def crop(img, crop_points):
    print(crop_points)
    x1, y1 = np.amin(crop_points, axis=0)
    x2, y2 = np.amax(crop_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    #min_points = np.amin(np.array(list(crop_points)), axis=0)
    #max_points = np.amax(np.array(list(crop_points)), axis=0)


    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

    eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect


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
        print(frame.shape)
        img_height, img_width, _ = frame.shape

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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


        # Face_Landmark Crop
        mesh_coords = landmarksDetection(frame, face_landmarks, False)
        left_eye_points = np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)
        right_eye_points = np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)
        nose_points = np.array([mesh_coords[p] for p in NOSE], dtype=np.int32)
        lip_points = np.array([mesh_coords[p] for p in LIP], dtype=np.int32)

        left_eye, left_eye_rect = crop(frame, left_eye_points)
        right_eye, right_eye_rect = crop(frame, right_eye_points)
        nose, nose_rect = crop(frame, nose_points)
        lip, lip_rect = crop(frame, lip_points)
        whole_face, whole_face_rect = crop(frame, mesh_coords)
        cv2.rectangle(frame, pt1=tuple(left_eye_rect[0:2]), pt2=tuple(left_eye_rect[2:4]), color=(255, 255, 255),
                      thickness=2)
        cv2.rectangle(frame, pt1=tuple(right_eye_rect[0:2]), pt2=tuple(right_eye_rect[2:4]), color=(255, 255, 255),
                      thickness=2)
        cv2.rectangle(frame, pt1=tuple(nose_rect[0:2]), pt2=tuple(nose_rect[2:4]), color=(255, 255, 255),
                      thickness=2)
        cv2.rectangle(frame, pt1=tuple(lip_rect[0:2]), pt2=tuple(lip_rect[2:4]), color=(255, 255, 255),
                      thickness=2)
        cv2.rectangle(frame, pt1=tuple(whole_face_rect[0:2]), pt2=tuple(whole_face_rect[2:4]), color=(255, 255, 255),
                      thickness=2)
        frame = cv2.flip(frame, 1)

        # 화면에 띄우기
        cv2.imshow('frame', annotated_image)
        cv2.imshow('frame2', frame)
        cv2.imshow('frame3', whole_face)

        # 'q'버튼 누르면 종료하기
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    video.release()
    cv2.destroyAllWindows()
