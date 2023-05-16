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
After capturing a single frame from the camera, it's initially processed using the Face Mesh model from MediaPipe, as shown in ConvNet 1 of Fig 1. From the resulting output, the mesh points corresponding to the face with the largest proportion in the frame is extracted. To exclude factors such as hairstyle, the face region was cropped using the mesh points, focusing only on the facial area.  
And to check whether a person is waering a mask or not, we use pretrained binary classification convnet from [here]()


<br>


## Reference
```

```
