# Mask Face Verification System using Deep Learning

With the advancement of deep learning and high-performance GPUs, it has become possible to perform face authentication using normal cameras. Moreover, the mandatory use of masks due to COVID-19 has posed challenges in utilizing functionalities like Face ID. So this project aims to create a Face ID authentication system using only normal cameras through deep learning, and to make Face ID authentication possible even when waering a mask.

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
<img width=100% height=100% src="https://github.com/jaejungscene/Mask-Face-Verification-on-Web/assets/88542073/dc20f396-bee9-4320-b57f-1654050de856">
</p>
<!-- <div style="margin-right: 30px; margin-left: 30px;"> -->
<dl><dd><dl><dd>
After capturing a single frame from the camera, it's initially processed using the Face Mesh model from MediaPipe, as shown in ConvNet 1 of Fig 1. From the resulting output, the mesh points corresponding to the face with the largest proportion in the frame is extracted. To exclude factors such as hairstyle, the face region was cropped using the mesh points, focusing only on the facial area.  
And to check whether a person is waering a mask or not, we use pretrained binary classification convnet from [here]()
</dd></dl></dd></dl>
<!-- </div> -->

### How to extract unique feature(embedding vector) that vary among individuals
<p align='center'>
  <img width=21% height=21% src="https://github.com/jaejungscene/Mask-Face-Verification-on-Web/assets/88542073/63df44ac-9212-41b7-b87a-ee8c57bab51d"><br>
  <img width=50% height=50% src="https://github.com/jaejungscene/Mask-Face-Verification-on-Web/assets/88542073/d98ff2ad-4bad-4040-b7e9-609e217ca419"><br>  
  <img width=50% height=50% src="https://github.com/jaejungscene/Mask-Face-Verification-on-Web/assets/88542073/9e09d8d3-2591-4584-9514-0a5c162be0ef">
</p>
<dl><dd><dl><dd>
We train the model using face recognition methods, such as ArcFace and CosFace, with the MS1MV2 dataset (containing 85K IDs and 5.8M images). The model employed is iresnet15, a ResNet-based architecture commonly used in the field of face recognition, producing a 512-dimensional vector as its output. During face recognition training, we utilize an fc layer with 85K classes from the MS1MV2 dataset. However, after training, we discard this fc layer and only use the ConvNet, which outputs the 512-dimensional vector.  
**We achieve the highest score (TAR@FAR) when using CosFace during training**.
</dd></dl></dd></dl>



<br>

## Reference
```

```
