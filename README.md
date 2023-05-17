# Mask Face Verification System using Deep Learning
<dl><dd><dl><dd>
With the advancement of deep learning and high-performance GPUs, it has become possible to perform face authentication using normal cameras. Moreover, the mandatory use of masks due to COVID-19 has posed challenges in utilizing functionalities like Face ID. So this project aims to create a Face ID authentication system using only normal cameras through deep learning, and to make Face ID authentication possible even when waering a mask.
</dd></dl></dd></dl>




<br>




## Overview of the system
### Demo
- [Demo Video Link](https://jaejung.notion.site/Demo-Mask-Face-Verification-1c0b94dbbbfc471e816825ec75bad35f)
### Abstraction of System Architecture & Workflow
![Screenshot 2023-05-16 at 4 09 54 PM](https://github.com/jaejungscene/Mask-Face-Verification-on-Web/assets/88542073/c2d41583-0d79-43c0-a548-22a0425f179e)




<br>




## Detail of The System


### How to determine whether a person is wearing a mask on their face or not & How to crop and align only one person's face from a frame
<p align='center'>
  <img width=94% height=94% src="https://github.com/jaejungscene/Mask-Face-Verification-on-Web/assets/88542073/dc20f396-bee9-4320-b57f-1654050de856">
  <br>
  <b>Fig 1</b>
</p>
<!-- <div style="margin-right: 30px; margin-left: 30px;"> -->
<dl><dd><dl><dd>
After capturing a single frame from the camera, it's initially processed using the Face Mesh model from MediaPipe, as shown in ConvNet 1 of Fig 1. From the resulting output, the mesh points corresponding to the face with the largest proportion in the frame is extracted. To exclude factors such as hairstyle, the face region was cropped using the mesh points, focusing only on the facial area.  
And to check whether a person is waering a mask or not, we use pretrained binary classification convnet from [here]()
</dd></dl></dd></dl>
<!-- </div> -->


### How to extract unique feature(embedding vector) that vary among individual faces
<p align='center'>
  <img width=21% height=21% src="https://github.com/jaejungscene/Mask-Face-Verification-on-Web/assets/88542073/63df44ac-9212-41b7-b87a-ee8c57bab51d"><br>
  <img width=50% height=50% src="https://github.com/jaejungscene/Mask-Face-Verification-on-Web/assets/88542073/d98ff2ad-4bad-4040-b7e9-609e217ca419"><br>  
  <img width=50% height=50% src="https://github.com/jaejungscene/Mask-Face-Verification-on-Web/assets/88542073/9e09d8d3-2591-4584-9514-0a5c162be0ef">
</p>
<dl><dd><dl><dd>
We train the model using face recognition methods, such as ArcFace and CosFace, with the MS1MV2 dataset (containing 85K IDs and 5.8M images). The model employed is iresnet15, a ResNet-based architecture commonly used in the field of face recognition, producing a 512-dimensional vector as its output. During face recognition training, we utilize an fc layer with 85K classes from the MS1MV2 dataset. However, after training, we discard this fc layer and only use the ConvNet, which outputs the 512-dimensional vector.  
**We achieve the highest score (TAR@FAR) when using CosFace during training**.
</dd></dl></dd></dl>


### How to extract unique feature that vary among individuals wearing mask
<dl><dd><dl><dd>
<i><b>Try 1)</b></i> We attempted to use various traditional computer vision descriptors to extract unique features from patterns near the eyes and eyebrows of individuals. Descriptors such as `Local Binary Pattern`, `ORB`, and others were employed. However, the verification scores (TAR@FAR) obtained from all these descriptors were significantly low, rendering them unsuitable for utilization in a real system (This limitation may be attributed to the similarity of human faces when viewed by traditional descriptors). Consequently, we have made the decision to exclude all traditional computer vision algorithms when extracting unique features from individuals wearing masks.
</dd></dl></dd></dl>
<p align='center'>
  <img src="https://github.com/jaejungscene/Mask-Face-Verification-on-Web/assets/88542073/020f9529-2b32-4192-b913-46045feb68e2">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/jaejungscene/Mask-Face-Verification-on-Web/assets/88542073/8ff8fb6b-935c-4b49-90b6-4340684d08cb">
  <br>
  &nbsp;&nbsp;
  <b>Fig 2.</b> Cutout
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>Fig 3.</b> Grad-CAM
</p>
<dl><dd><dl><dd>
<i><b>Try 2)</b></i> We attempted to utilize deep learning for our task. To prevent the deep learning model (ConvNet 0) from solely focusing on the mask, which exhibits a wide range of colors and shapes, we applied `Cutout` processing to the regions where the presence of a mask was likely, as depicted in Fig 2. Subsequently, we retrained the model. Through experimentation using `Grad-CAM`, as illustrated in Fig 3, we verified that the model concentrated on the areas surrounding the eyes and eyebrows. Also, we achieved favorable performance scores (TAR@FAR) when deploying the model in real-world scenarios. As a result, we have decided to adopt this approach for extracting unique features when individuals are wearing a mask.
</dd></dl></dd></dl>


### How to find the optimal threshold value
<pre>
FAR = (FP/(FP+TN)) * 100  #Critical for security, should converge to 0.
TAR = (TP/(TP+FN)) * 100  #The higher it is, the higher the success rate of Face ID.
</pre>
<p align='center'>
  <img width=90% height=90% src="https://github.com/jaejungscene/Mask-Face-Verification-on-Web/assets/88542073/bbb7ad20-2e56-4fbb-98ab-6ea3f380e755">
  <br>
  <b>Example of threshold search algorithm</b>
</p>
<dl><dd><dl><dd>
If the threshold value for similarity is too high, the failure rate of Face ID for most individuals will increase. On the contrary, if it is set too low, there is a critical risk of other people's faces passing the Face ID authentication. Therefore, it is important to find an appropriate threshold value that can lower the failure rate of Face ID while ensuring security. To achieve this, we utilized the LFW dataset, which contains images similar to real-life scenarios, and initially set the threshold value to 0. Then, we gradually increased the threshold value by 0.1 while observing the False Acceptance Rate (FAR) converging to 0 and the True Acceptance Rate (TAR) reaching its highest point. Once we identified the optimal threshold value at the first decimal place, we set the starting threshold value to the optimal value minus 0.1 and repeated the process to find the optimal threshold value in the same manner. We continued this iteration until we found the optimal threshold value at the third decimal place.
</dd></dl></dd></dl>

<br>

## Reference
- [FaceNet: A Unified Embedding for Face Recognition and Clustering. Florian Schroff, Dmitry Kalenichenko, James Philbin](https://arxiv.org/pdf/1503.03832.pdf)
- [SphereFace: Deep Hypersphere Embedding for Face Recognition. Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj, Le Song](https://arxiv.org/pdf/1704.08063.pdf)
- [CosFace: Large Margin Cosine Loss for Deep Face Recognition. Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou, Zhifeng Li, Wei Liu](https://arxiv.org/pdf/1801.09414.pdf)
- [ArcFace: Additive Angular Margin Loss for Deep Face Recognition. Jiankang Deng, Jia Guo, Jing Yang, Niannan Xue, Irene Kotsia, Stefanos Zafeiriou](https://arxiv.org/pdf/1801.07698.pdf)
- [VGGFace2: A dataset for recognising faces across pose and age. Qiong Cao, Li Shen, Weidi Xie, Omkar M. Parkhi, Andrew Zisserman](https://arxiv.org/pdf/1710.08092.pdf)
- [MS-Celeb-1M: A Dataset and Benchmark for Large-Scale Face Recognition. Yandong Guo, Lei Zhang, Yuxiao Hu, Xiaodong He, Jianfeng Gao](https://arxiv.org/pdf/1607.08221.pdf)
- [https://github.com/seriousran/face-recognition](https://github.com/seriousran/face-recognition)
- [https://github.com/chandrikadeb7/Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection)
