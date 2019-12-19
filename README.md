# Fase Swapping


## Step-by-step solution
### 1. Face detection
![step_1](./imgs/step_1.png)

### 2. Keypoints detection
![step_2](./imgs/step_2.png)

### 3. Face cropping
![step_3](./imgs/step_3.png)

### 4. Face triangulation
![step_4](./imgs/step_4.png)

### 5. Reconstruction new face
![step_5_idea](./imgs/step_5.png)
![step_5_gif](./imgs/step_5.gif)

### 6. Pull up new face
![step_5_end](./imgs/step_6.png)

### 6. Seamless Cloning
![step_5_end](./imgs/step_7.png)


## Techology part
The most of the algorithms are written using [openCV](https://pypi.org/project/opencv-python/)

* Face detection: **Haar Cascade**
* Keypoint detection: **The cascade of regressors** using [dlib](http://dlib.net) based on this [paper](http://www.nada.kth.se/~sullivan/Papers/Kazemi_cvpr14.pdf)
* Face triangulation: simple **Triangulation**
* Reconstruction: **Affine projection** each triangles 
* Color alignment: **Seamless Cloning**
