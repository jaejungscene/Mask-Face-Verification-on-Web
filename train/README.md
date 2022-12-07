# Used Dataset
- Training & Validation Dataset
  - MS1MV2 (IDs: 85K, Total Images: 5.8M) => [download site](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-arcface-85k-ids58m-images-57)
- Validation Dataset
  - 모든 LFW 이미지를 validation dataset으로 활용하려고 했으나 arcface, cosface margin loss를 활용한 모델 특정상 backbone과 fc-layer가 분리되어 있어 다른 fc-layer가 필요함으로 쓰지 못 함. 
- Test Dataset (in order to find threshold for verification)
  - LFW with more than one Image (IDs: 1.68K, Total Images: 9.16K) => [download site](http://vis-www.cs.umass.edu/lfw/#download)
  - Face Mask Dataset Generated GAN (FMDG) in kaggle => [download site](https://www.kaggle.com/datasets/prasoonkottarathil/face-mask-lite-dataset)
  - Asian Face Mask Dataset (AFMD) in kaggle => [download site](http://vis-www.cs.umass.edu/lfw/#download)


<br>
<br>

# Result

| model(#param) | Loss function | LFW | FMDG | NFMD
|---|---|---|---|---|
| resnet50(25M)|
|iresnet18(24M)| arcface() | 0.92 | 0.209433 | 0.5 |
|iresnet18(24M)| arcface() | 0.92 | 0.209433 | 0.5 |
|iresnet18(24M)| ***arcface()*** | ***0.92*** | ***0.209433*** | ***0.5*** &nbsp; <= &nbsp; best |
|iresnet18(24M)| cosface() | 0.92 | 0.209433 | 0.5 |


