# Used Dataset
- Training Dataset
  - MS1MV2 (IDs: 85K, Total Images: 5.8M) => [download site](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-arcface-85k-ids58m-images-57)
- Validation Dataset
  - All of LFW (IDs: 5K, Total Images: 13K) => [download site](http://vis-www.cs.umass.edu/lfw/#download)
- Test Dataset (in order to find threshold for verification)
  - LFW with more than one Image (IDs: 1.68K, Total Images: 9.16K)
  - Face Mask Dataset Generated GAN (FMDG) in kaggle => [download site](https://www.kaggle.com/datasets/prasoonkottarathil/face-mask-lite-dataset)
  - Natural Face Mask Dataset (NFMD) in kaggle => [download site](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)


<br>
<br>

# Result

| Loss function | LFW | FMDG | NFMD
|---|---|---|---|
| arcface() | 0.92 | 0.209433 | 0.5 |
| arcface() | 0.92 | 0.209433 | 0.5 |
| ***arcface()*** | ***0.92*** | ***0.209433*** | ***0.5*** &nbsp; <= &nbsp; best |
| cosface() | 0.92 | 0.209433 | 0.5 |


