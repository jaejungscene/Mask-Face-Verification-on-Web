import cv2
import math
import numpy as np
import mediapipe as mp
import argparse
import os
from torchvision import transforms
import torch
from model import get_model

IMG_SIZE = (200, 200)
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
NOSE = [6, 122, 188, 114, 129, 98, 97, 2, 326, 327, 358, 343, 412, 351]
LIP = [0, 37, 39, 61, 84, 17, 314, 291, 267]


class face_vectorization:
    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        # mediapipe configs
        self.mp_face_mesh = mp.solutions.face_mesh

        # 모델 불러오기
        self.model = get_model(self.base_dir)
        self.model.eval()

    def landmarksDetection(self, img, face_landmarks, draw=False):
        img_height, img_width = img.shape[:2]
        
        mesh_coord = [[int(point.x * img_width), int(point.y * img_height)] for point in
                      face_landmarks.landmark]
        if draw:
            [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]
            print(1)

        # returning the list of tuples for each landmarks
        return mesh_coord

    def crop(self, img, crop_points):
        x1, y1 = np.amin(crop_points, axis=0)
        x2, y2 = np.amax(crop_points, axis=0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        #w = (x2 - x1) * 1.2
        #h = w * IMG_SIZE[1] / IMG_SIZE[0]
        w = (x2 - x1) * 1.2
        h = w * IMG_SIZE[1] / IMG_SIZE[0]

        margin_x, margin_y = w / 2, h / 2

        min_x, min_y = int(cx - margin_x), int(cy - margin_y)
        max_x, max_y = int(cx + margin_x), int(cy + margin_y)

        rect = np.rint([min_x, min_y, max_x, max_y]).astype(int)

        img = img[rect[1]:rect[3], rect[0]:rect[2]]

        return img, rect

    def save_img(self, np_img, path, model):
        print("-" * 50)
        print("saving image")
        #np_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if np_img.shape[-1] > 3:
            print("remove alpha channel")
            np_img = np_img[:, :, 0:3]
        print(np_img.shape)

        min_size = min(np_img.shape[0:-1])

        image_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((min_size, min_size)),
            transforms.Resize(size=(112, 112)),
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

        np.save(os.path.join(self.base_dir, path), output.numpy())
        return output

    def capturing(self, userId):
        # 웹캠으로 테스트
        video = cv2.VideoCapture(0)
        emb = None

        with self.mp_face_mesh.FaceMesh(
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

                for face_landmarks in results.multi_face_landmarks:
                    pass

                # Face_Landmark Crop
                mesh_coords = self.landmarksDetection(frame, face_landmarks, False)
                left_eye_points = np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)
                right_eye_points = np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)
                nose_points = np.array([mesh_coords[p] for p in NOSE], dtype=np.int32)
                lip_points = np.array([mesh_coords[p] for p in LIP], dtype=np.int32)

                whole_face, whole_face_rect = self.crop(frame, mesh_coords)

                # 전처리된 이미지 보려면 주석 해제
                squeezed_face = whole_face.squeeze() # 전처리된 얼굴 이미지
                cv2.imshow('preprocessed_face', squeezed_face)

                # 'q'버튼 누르면 npy 저장 후 종료하기
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    emb = self.save_img(squeezed_face, "images/"+str(userId)+".npy", self.model)
                    #print(type(squeezed_face), type(frame))
                    break

            video.release()
            cv2.destroyAllWindows()

        return emb

if __name__ == "__main__":
    model = face_vectorization()
    emb = model.capturing()
    print(emb)
