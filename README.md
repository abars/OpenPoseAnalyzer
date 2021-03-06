# Analyze OpenPose Output

# Overview

## Requirements

Caffe

OpenCV

Python 2.7

# Test

## Download Pretrained-Model

`python download_model.py`

## Display OpenPose Output

`python display_openpose_output.py`

# OpenPose Architecture

## OpenPose Output

* PAF (1, 38, 46, 46) - output[0] 

<img src="https://github.com/abars/OpenPoseAnalyzer/blob/master/images/paf.png" width="50%" height="50%">

* CONFIDENCE (1, 19, 46, 46) - output[1]

<img src="https://github.com/abars/OpenPoseAnalyzer/blob/master/images/confidence.png" width="50%" height="50%">

## COCO KeyPoint

<img src="https://github.com/abars/OpenPoseAnalyzer/blob/master/images/keypoint.png" width="50%" height="50%">
(image from wider face dataset)


* Nose – 0
* Neck – 1
* Right Shoulder – 2
* Right Elbow – 3
* Right Wrist – 4
* Left Shoulder – 5
* Left Elbow – 6
* Left Wrist – 7
* Right Hip – 8
* Right Knee – 9
* Right Ankle – 10
* Left Hip – 11
* Left Knee – 12
* LAnkle – 13
* Right Eye – 14
* Left Eye – 15
* Right Ear – 16
* Left Ear – 17
* Background – 18

# Related Work

<https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/>

<https://github.com/kevinlin311tw/keras-openpose-reproduce>

<https://github.com/ArashHosseini/3d-pose-baseline/>
