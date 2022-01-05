# GAMNet
# GAMNet: Global Attention Via Multi-scale Context for Binocular Depth Estimation Algorithm and Application
## Abstract
Deep neural networks significantly enhance the accuracy of the stereo-based disparity estimation.
Some current methods exist inefficient use of the global context information, which will lead to the loss of structural details in ill-posed areas. To this end, we design a novel stereo network GAMNet composed of the three core components (GDA, MPF, DCA) for estimating the depth prediction in challenging real-world environments. Firstly, we present a lightweight attention module, integrating the global semantic cues for every feature position across the channel and spatial dimensions. Next, we construct the MPF module to encode and fuse the diverse semantic and contextual information from different levels of the feature pyramid. Finally, we aggregate cost volume with a stacked encoder-decoder composed of our DCA module and 3D convolutions, filtering the transmission of matching clues and capturing the rich global contexts. Substantial experiments conducted on KITTI 2012, KITTI 2015, SceneFlow, and Middlebury-v3 datasets manifest that GAMNet surpasses preceding methods with contour-preserving disparity predictions. 
In addition, we first try to propose a 3D scene reconstructions evaluation strategy (a spatial grasping point linear evaluation strategy) for the end-to-end stereo networks in an unsupervised mode, which deploys on our designed robot vision-guided system. In application experiments, our method can produce densely high-precision 3D reconstructions to implement the grasping task in complex real-world scenes and achieves excellent robust performance with competitive inference efficiency.
## Installation
The code was tested with Ubuntu20.04 with Anaconda environment:
* CUDA 11.1
* python 3.7
* pytorch 11.0
* opencv-python 4.5.1
## Training
* bash save.sh
## Video
* 3D vision-guided system based on GAMNet
https://www.bilibili.com/video/BV19r4y1U7oR/
