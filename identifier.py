import cv2
import math
import numpy as np
from data_loader import masked_face_dataset, ImageTransform
import mediapipe as mp
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet18
from torchvision import transforms
from FaceNet import FaceNet_ResNet18
from model import get_model
import os

base_dir = "C:/Users/gmk_0/source/repos/pythonProject/IT2/Computer-Vision-Project"
model = get_model()
model.eval()
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
IMG_SIZE = (200, 200)
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
NOSE = [6, 122, 188, 114, 129, 98, 97, 2, 326, 327, 358, 343, 412, 351]
LIP = [0, 37, 39, 61, 84, 17, 314, 291, 267]
num_of_frames = {"Mask": 0, "No Mask": 0}
embeddings = []
matched = 0


# mediapipe configs
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
                help="path to trained face mask detector model")
args = vars(ap.parse_args())



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
    x1, y1 = np.amin(crop_points, axis=0)
    x2, y2 = np.amax(crop_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    rect = np.rint([min_x, min_y, max_x, max_y]).astype(int)

    img = img[rect[1]:rect[3], rect[0]:rect[2]]

    return img, rect

def load_img():
    print("-"*50)
    print("loading image")
    np_img = cv2.imread(os.path.join(base_dir, f"images/2.jpg"))
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    if np_img.shape[-1] > 3:
        print("remove alpha channel")
        np_img = np_img[:, :, 0:3]
    print(np_img.shape)
    
    max_size = max(np_img.shape[0:-1])
    min_size = min(np_img.shape[0:-1])

    image_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((min_size, min_size)),
        transforms.Resize(size=(112,112)), 
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225]
        )
    ])

    torch_img = image_transforms(np_img)
    C, H, W = torch_img.size()
    torch_img = torch_img.view(1, C, H, W)
    print('input shape', torch_img.shape)
    with torch.no_grad():
        model.eval()
        output = model(torch_img)
        
    print("output shape: ", output.shape)
    return output

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

def predict_mask(face, model):
    # Image Preprocessing
    #face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_1 = cv2.resize(face, dsize=(224, 224))
    face_2 = cv2.resize(face, dsize=(112, 112))
    face_1 = img_to_array(face_1)
    face_2 = img_to_array(face_2)
    face_1 = face_1.reshape((1, 224, 224, 3))
    face_1 = preprocess_input(face_1)
    pred = model.predict(face_1, batch_size=1)
    face_2 = face_2.reshape((1, 112, 112, 3))

    return pred, face_2

def cos_sim(x:torch.tensor, y:torch.tensor)->torch.tensor:
    return x@y.T / torch.norm(x)*torch.norm(y)


# 모델 불러오기
maskNet = load_model(args["model"])

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
        img_height, img_width, _ = frame.shape

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Draw face landmarks of each face.

        if not results.multi_face_landmarks:
            continue
        annotated_image = frame.copy()
        cropped_image = frame.copy()

        for face_landmarks in results.multi_face_landmarks:
            pass
            """mp_drawing.draw_landmarks(
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
                    .get_default_face_mesh_iris_connections_style())"""


        # Face_Landmark Crop
        mesh_coords = landmarksDetection(cropped_image, face_landmarks, False)
        left_eye_points = np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)
        right_eye_points = np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)
        nose_points = np.array([mesh_coords[p] for p in NOSE], dtype=np.int32)
        lip_points = np.array([mesh_coords[p] for p in LIP], dtype=np.int32)

        whole_face, whole_face_rect = crop(frame, mesh_coords)
        #frame = cv2.flip(frame, 1)

        # pred : 마스크 감지 모델 결과
        pred, face = predict_mask(whole_face, maskNet)
        pred = pred.squeeze()
        [mask, withoutMask] = pred
        #print("Mask: ", mask, "Without mask: ", withoutMask)

        label = "Mask" if mask > withoutMask else "No Mask"
        if label == "Mask":
            num_of_frames["Mask"] += 1
            num_of_frames["No Mask"] = 0
            color = (0, 255, 0)
        else:
            num_of_frames["No Mask"] += 1
            num_of_frames["Mask"] = 0
            color = (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (whole_face_rect[0], whole_face_rect[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, pt1=tuple(whole_face_rect[0:2]), pt2=tuple(whole_face_rect[2:4]), color=color,
                      thickness=2)

        # No Mask / Mask 일정 프레임 도달시
        if num_of_frames["No Mask"] >= 5:

            face = torch.from_numpy(face)
            print(face.shape)
            face = torch.permute(face, [0, 3, 1, 2])  # 모델에 집어넣기 위한 전처리
            with torch.no_grad():
                embedding = model(face).squeeze()
            print("embedding: ", len(embedding)) #128
            
            loaded_image = load_img()
            result = cos_sim(embedding, loaded_image)
            print("result", result)
            if result > 80:
                matched += 1
                if matched > 7: break


        elif num_of_frames["Mask"] >= 5:
            left_eye, left_eye_rect = crop(cropped_image, left_eye_points)
            right_eye, right_eye_rect = crop(cropped_image, right_eye_points)
            nose, nose_rect = crop(cropped_image, nose_points)
            lip, lip_rect = crop(cropped_image, lip_points)
            cv2.rectangle(cropped_image, pt1=tuple(left_eye_rect[0:2]), pt2=tuple(left_eye_rect[2:4]),
                          color=(255, 255, 255),
                          thickness=2)
            cv2.rectangle(cropped_image, pt1=tuple(right_eye_rect[0:2]), pt2=tuple(right_eye_rect[2:4]),
                          color=(255, 255, 255),
                          thickness=2)
            cv2.rectangle(cropped_image, pt1=tuple(nose_rect[0:2]), pt2=tuple(nose_rect[2:4]), color=(255, 255, 255),
                          thickness=2)
            cv2.rectangle(cropped_image, pt1=tuple(lip_rect[0:2]), pt2=tuple(lip_rect[2:4]), color=(255, 255, 255),
                          thickness=2)
            cv2.imshow('cropped_image', cropped_image)


        # 화면에 띄우기
        # cv2.imshow('frame', annotated_image)
        cv2.imshow('frame', frame)
        print(cropped_image.shape, face.shape)
        # 전처리된 이미지 보려면 주석 해제
        #squeezed_face = face.squeeze() # 전처리된 얼굴 이미지
        #cv2.imshow('preprocessed_face', squeezed_face)

        print(num_of_frames)


        # 'q'버튼 누르면 종료하기
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    video.release()
    cv2.destroyAllWindows()
