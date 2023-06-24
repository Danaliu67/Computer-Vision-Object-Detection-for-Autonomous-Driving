# Computer Vision: Object Detection for Autonomous Driving

This project investigates techniques for improving the detection performance of an autonomous driving detection system. The study systematically adjusts training epochs, batch size, and input image size on the ResNet network and compares and explains these parameters. Results demonstrate that ResNet101 outperforms ResNet50 and increasing the image size can significantly improve mean average precision. Additionally, suitable batch sizes must be selected to balance time and performance, and a suitable number of epochs should be chosen based on practical situations to achieve optimal performance. The findings provide valuable guidance for the study and practice of autonomous driving, and suggest the need for further exploration of other factors that affect object detection performance to propose more efficient and accurate system design methods.

## 1. Introduction
[**ResNet**](https://arxiv.org/abs/1512.03385) and [**YOLO**](https://arxiv.org/abs/1506.02640) are popular algorithms in autonomous driving detection systems, with ResNet solving the issue of gradient vanishing and YOLO providing fast and accurate real-time object detection. 

Recent studies have focused on combining these two methods to improve detection performance[1], with adjustments to training epochs, batch size, and image size showing promise in enhancing system capabilities for detecting small objects[2]. Our report aims to investigate these techniques to optimize detection performance.


## 2. Experiment
### 2.1 Experimental data
This 2D image dataset for autonomous driving object detection is created by HUAWEI and contains 5K labeled training images, 2.5K labeled validation images, and 2.5K images with hidden annotations. It consists of 5 classes: 'Pedestrian', 'Cyclist', 'Car', 'Truck', and 'Tram', with their distribution depicted in a figure.

<!-- ![image](resources/image.png) -->
<div align="center">
  <img src="./resources/table1.jpg" height="300">
</div>
<p align="center">
  Table 1. Distribution of Classes in Training and Validation Sets.
</p>

It is clear that the training and validation data sets show an uneven distribution, with over 50% of the data in the 'car' category and only 4% in the 'tram' category. This results that in the subsequent target detection, the detection performance (AP) for 'car' is better and for 'tram' is poorer , which is mainly attributed to the data distribution rather than the detection model itself.

Upon examining the images provided by the dataset, it was observed that the majority of the detection targets in the images are relatively small, with fewer large objects present. This observation can have implications on the selection of parameters in the subsequent object detection process, such as avoiding excessive down sampling.

### 2.2 Experimental Setting

Our experiment design comprised variations in backbone network, epoch number, batch size, and image size, with the aim of providing insights into the optimal configuration for achieving optimal object detection performance.

<div align="center">

| Different Techniques | Setting                |
|----------------------|------------------------|
| Backbone Networks    | ResNet50 and ResNet101 |
| Epochs               | 24 and 48              |
| Batch Sizes          | 12, 16, 24, and 32     |
| Image Sizes          | 448, 512, and 640      |

</div>

<p align="center">
  Table 2. Parameter Setting
</p>

We employed the Adam optimizer with an initial learning rate of 0.001, which gradually decayed as training progressed.

### 2.3 Experimental Results

We conducted experiments on the parameter settings mentioned above, aiming to improve the object detection performance. In this sections, we will present the detailed training configurations and results of these variations.

During training, we recorded the loss curve below(ls: image size, bs: batch size):

<!-- ![image](resources/image.png) -->
<div align="center">
  <img src="./resources/table1.jpg" height="300">
</div>
<p align="center">
  Figure 1. Loss Curve (ls=448, bs=12).
</p>

<!-- ![image](resources/image.png) -->
<div align="center">
  <img src="./resources/table1.jpg" height="300">
</div>
<p align="center">
  Figure 1. Loss Curve(ls=448, bs=32).
</p>

<!-- ![image](resources/image.png) -->
<div align="center">
  <img src="./resources/table1.jpg" height="300">
</div>
<p align="center">
  Figure 2. Distribution of Classes in Training and Validation Sets.
</p>

<!-- ![image](resources/image.png) -->
<div align="center">
  <img src="./resources/table1.jpg" height="300">
</div>
<p align="center">
  Figure 3. Loss Curve(ls=512, bs=12).
</p>

<!-- ![image](resources/image.png) -->
<div align="center">
  <img src="./resources/table1.jpg" height="300">
</div>
<p align="center">
  Figure 4. Loss Curve(ls=640, bs=12).
</p>

From the loss curve, we observed some trends:

1.	Initially, the model's performance metrics changed rapidly, but as training progressed, the changes slowed down, eventually converging. This may be because the model learned some basic features in the early epochs and more fine-grained features in later epochs.

2.	We found that the model trained for 24 epochs was slightly underfitting. However, after 48 epochs, the performance metrics reached a relatively stable level, indicating that the model had learned enough features from the training data.

3.	We found that the model's performance metrics gradually converged during training but exhibited some jitter to some extent. We attributed this to the noise and randomness in the training data.

We trained multiple models with different configurations and evaluated their performance using mAP. Results are presented in the following table.

<!-- ![image](resources/image.png) -->
<div align="center">
  <img src="./resources/table1.jpg" height="300">
</div>
<p align="center">
  Table 3. Resnet 50 under different parameter setting.
</p>

<div align="center">
  
|            | Image size = 640 |
|------------|------------------|
| Batch size |     12   |   32   |
| Epoch=24   |  0.50    | 0.49   |
  
</div>

<p align="center">
  Table 4. Resnet 101 under different parameter setting
</p>

<!-- ![image](resources/image.png) -->
<div align="center">
  <img src="./resources/table1.jpg" height="300">
</div>
<p align="center">
  Figure 5. Effect of Batch Size and Image Size on mAP: A Side-by-Side Comparison.
</p>

<!-- ![image](resources/image.png) -->
<div align="center">
  <img src="./resources/table1.jpg" height="300">
</div>
<p align="center">
  Figure 6. Longitudinal Comparison of Epochs on mAP.
</p>

From the chart, it can be observed that: 

1.	Increasing epoch numbers can improve mAP value, with the effect varying by image size and batch size. A limited improvement was observed when increasing epoch numbers from 24 to 48 for 448x448 and 512x512 image sizes, while a more significant improvement was observed for 640x640 images.
	
2.	Batch size had a minor effect on accuracy, with only slight differences observed between different image sizes. For the 448x448 image size, increasing batch size slightly decreased accuracy, while for the 512x512 and 640x640 image sizes, accuracy remained relatively stable.
  
3.	Image size was positively correlated with mAP value, indicating that increasing image size provides more detailed information about the image, leading to improved performance. The mAP value increased from 0.40 to 0.43 as image size increased from 448 to 512, with a further increase to 640 resulting in a more significant improvement, reaching 0.48.
	
4.	Network architecture played a crucial role in determining model performance. Resnet101 outperformed Resnet50 for the same parameters, with the highest mAP value of 0.50 obtained with Resnet101 for an image size of 640 and batch size of 12, while the mAP value for Resnet50 was 0.49.

### 2.4 Experimental Discussion

Our experiments demonstrate that the model learns basic features first and more complex features later, emphasizing the importance of adequate training epochs. We found that the model's performance was slightly underfitting at 24 epochs, but stable at 48 epochs, highlighting the need to balance underfitting and overfitting.

We noted that the dataset contains many small objects, making them difficult to detect with down-sampled images of 448. Increasing the image size to 640 improved the model's performance by preserving more details, although we should consider that it also increases model complexity and training difficulty. Our experiment identified an optimal image size of 640, providing valuable insights for future studies.

Increasing the batch size can potentially increase training efficiency but may introduce noise and negatively impact accuracy. Additionally, larger batch sizes may require more memory and computational resources, negatively impacting training speed.

Our experiment also showed that Resnet101 outperformed Resnet50 in terms of mAP for all parameter settings. However, this performance gain comes at the cost of increased computational complexity and training time. Thus, the choice of architecture should be based on the tradeoff between performance and computational resources.

### 2.5 Experimental Presentation

Based on the conducted experiment, the final object detection results are presented in the below Figure, which demonstrates a high detection performance. The detection algorithm successfully identified pedestrians and bicycles that are not easily discernible by the human eye. The results also showed robust detection performance in low-light conditions during the night.

<!-- ![image](resources/image.png) -->
<div align="center">
  <img src="./resources/table1.jpg" height="300">
</div>
<p align="center">
  Figure 7. Visualization of Experimental Detection Results.
</p>

The achieved detection performance can be attributed to the use of advanced object detection algorithms and techniques, which enables accurate object recognition and localization in complex scenes and challenging lighting conditions.

## 3. Conclusion

In summary, our study investigated techniques to improve the detection performance of an autonomous driving detection system. Our experiments highlighted the superiority of ResNet101 over ResNet50 and the significant impact of image size on mean average precision. We also found that selecting appropriate batch sizes is crucial for balancing time and performance. Our study provides valuable guidance for the study and practice of autonomous driving, emphasizing the importance of balancing underfitting and overfitting while considering the tradeoff between performance and computational resources. Further exploration of other factors affecting object detection performance is necessary to propose more efficient and accurate system design methods.


## References:
[1]	X. Xie et al., "Dense Convolutional Network Combined with YOLOv2 for Vehicle Detection in Aerial Images," Remote Sensing, vol. 10, no. 12, 2018.
[2]	S. Kim et al., "Improving the Performance of Object Detection Systems for Autonomous Vehicles Using Deep Learning," Electronics, vol. 10, no. 2, 2021.
