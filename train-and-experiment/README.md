# Used Dataset
- Training & Validation Dataset
  - MS1MV2 (IDs: 85K, Total Images: 5.8M) => [download site](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-arcface-85k-ids58m-images-57)
- Test Dataset (in order to find threshold for verification)
  - LFW with more than one Image (IDs: 1.68K, Total Images: 9.16K) => [download site](http://vis-www.cs.umass.edu/lfw/#download)
  - Face Mask Dataset Generated GAN (FMDG) in kaggle => [download site](https://www.kaggle.com/datasets/prasoonkottarathil/face-mask-lite-dataset)


<br>
<br>


# Training
- train 방법
  1) train.py의 ROOT_DIR을 train-and-experiment만 제외하고 바꿈 
  2) `sh train.sh` 명령어 입력 (train.sh 안에 있는 command를 통해서 margin 변환 가능
- train시 전체 모델은 embedding vector를 만들어 내는 `core backbone network`와 train을 학습하기 위한 `fc-layer`가 따로 존재.
- reference  
&nbsp;&nbsp;&nbsp; [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)  
&nbsp;&nbsp;&nbsp; [(*)Sphereface: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/pdf/1704.08063.pdf)  
&nbsp;&nbsp;&nbsp; [(**)Cosface: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.09414.pdf)  
&nbsp;&nbsp;&nbsp; [(***)Arcface: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698.pdf)  


<br>
<br>


# Search Threshold
- [code](https://github.com/jaejungscene/Computer-Vision-Project/blob/main/train-and-experiment/search-threshold.py)
- 가장 좋은 모델로 test 데이터들에 대한 threashold를 search함.  
- TAR@FAR<0.0001% Metric으로 search함 (FAR가 0.0001%를 넘지 않으면서 TAR가 가장 높은 것을 search).
- TAR : True Acceptance Rate = Recall = TP/(TP+FN).  
- FAR : False Acceptane Rate = Fall-out = FP/(FP+TN).  


<br>
<br>


# Additional Verification Methods for Masked Face
- [code](https://github.com/jaejungscene/Computer-Vision-Project/blob/main/train-and-experiment/test-descriptor.ipynb)
- LBP descriptor (Local Binary Pattern)
- ORB descriptor (Oriented FAST and Rotated BRIEF)
