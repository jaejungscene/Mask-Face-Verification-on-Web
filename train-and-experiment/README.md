# Used Dataset
- Training & Validation Dataset
  - MS1MV2 (IDs: 85K, Total Images: 5.8M) => [download site](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-arcface-85k-ids58m-images-57)
  - asian face를 사용하려고 했으나 중국 drive 사이트 Baidu에서 다운 받아야 하기에 하지 않음.
- Test Dataset (in order to find threshold for verification)
  - LFW with more than one Image (IDs: 1.68K, Total Images: 9.16K) => [download site](http://vis-www.cs.umass.edu/lfw/#download)
  - Face Mask Dataset Generated GAN (FMDG) in kaggle => [download site](https://www.kaggle.com/datasets/prasoonkottarathil/face-mask-lite-dataset)
  - Asian Face Mask Dataset (AFMD) in kaggle => [download site](http://vis-www.cs.umass.edu/lfw/#download)


<br>
<br>

# Result
- Model  
Pretrined resnet50 on ImageNet  
Pretrined iresnet18 on MS1MV3
- 아이디어, 학습 방법 및 margin loss들은 밑 논문들을 통해 구현 및 차용  
&nbsp;&nbsp;&nbsp; [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)  
&nbsp;&nbsp;&nbsp; [(*)Sphereface: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/pdf/1704.08063.pdf)  
&nbsp;&nbsp;&nbsp; [(**)Cosface: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.09414.pdf)  
&nbsp;&nbsp;&nbsp; [(***)Arcface: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698.pdf)  

| model(#param) | Loss function(margin size) | Accuracy
|---|---|---|
| resnet50(25M)| softmax | 1%(성능이 안나와 중간에 중단)
|iresnet18(24M)| arcface(0.5) | 13.7% (성능이 안나와 중간에 중단)
|iresnet18(24M)| arcface(0.5) + cosface(0.35) | 2.3% (성능이 안나와 중간에 중단)
|iresnet18(24M)| ***cosface(0.35)*** | ***85.7%*** |

<br>
<br>

# Search Threshold
- 가장 좋은 모델로 test 데이터들에 대한 threashold를 search함.  
- TAR@FAR=0.01% Metric으로 search함.
- TAR : True Acceptance Rate = Recall = TP/(TP+FN).  
- FAR : False Acceptane Rate = Fall-out = FP/(FP+TN).  
- TAR@FAR=0.01% : FAR를 0.01%로 fix하였을 때 TAR의 값.  

- FAR가 0.001%로 유지되면서 동시에 TAR의 값이 가장 높은 threshold saerch.  

1.마스크를 쓰지 않은 경우 0.36 ~ 0.37이 적정  

| threshold | TAR@FAR <- average value |
| -- | -- |
| 0.35 | 77.47% @ (FAR < 0.01%) |
| 0.36 | 75.48% @ (FAR < 0.0001%) |
| 0.37 | 73.04% @ (FAR < 0.0001%) |

2.마스크를 썼을 경우 0.36 ~ 0.37이 적정  

| threshold | TAR@FAR <- average value |
| -- | -- |
| 0.35 | 77.47% @ (FAR < 0.01%) |
| 0.36 | 75.48% @ (FAR < 0.0001%) |
| 0.37 | 73.04% @ (FAR < 0.0001%) |

### Problem about threshold search on test set
- test set으로 threshold를 search 하더라도 실시간 처리 시에는 컴퓨터에 따른 이미지의 domain 변화로 여기서 찾은 threshold가 유의미하지 않은 경우가 많이 발생한다.

<br>
<br>

# Additional Verification Methods for Mask Face
## 1. LBP descriptor (Local Binary Pattern)
## 2. ORB descriptor (Oriented FAST and Rotated BRIEF)
둘 다 눈 주의만으로 feature를 뽑아 냈을 시에 서로 다른 사람과의 feature의 차이가 거의 존재하지 않는다. 딥러닝 모델만으로 하는 것이 더 빠르고 좋을 것이라 판단된다.