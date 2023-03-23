# Using ResNet-50 for age and gender recognition.
# Trained on IMDB-WIKI dataset (500k+ images)

**Utils** folder contains preprocessing files: Dataset balancing, Face detection, Face cropping.

To run the demo:

1. download models folder from the link: https://drive.google.com/drive/folders/1_uEWVO26XJ1D_ey2Ilt269xT9hkIr9BL?usp=sharing
2. download utils folder (haar-cascade) 
3. run demo.py

![](https://github.com/AssanaliAbu/Age_Gender_recognition/blob/main/demo.gif)




# Preprocessing:
Dataset images were needed to be cropped down to face before training stage 

![](https://github.com/AssanaliAbu/Age_Gender_recognition/blob/main/images/crop_github.png)



**Also need to balance the dataset:**


![](https://github.com/AssanaliAbu/Age_Gender_recognition/blob/main/images/gender_dist_github.png)


![](https://github.com/AssanaliAbu/Age_Gender_recognition/blob/main/images/age_dist_github.png)


# Training

**Used pre-trained ResNet50 CNN model for both age and gender recognition.**
![](https://github.com/AssanaliAbu/Age_Gender_recognition/blob/main/images/resnet50-git.png)
