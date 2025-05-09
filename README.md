# Car Detection Using YOLOv5 using stanford dataset
exp_final: contain result of the training data

dataset used for training: yolov5\data\stanford.yaml


# Confusion matrix
![confusion_matrix](https://github.com/user-attachments/assets/27f9c91a-bfa1-418c-b6c3-8496ee9200a5)

The result of the images training is swohn in this confusion matrix. This training is based on the 195 model car.

# Augmetation
/augmentation/images: contain images that has been augmented by random hue value
/augmentation/labels: contain .txt file that have information of bounding box for each 195 classes

# Demo Video
https://youtu.be/B2h0slHK-S0

# Web Deployment
Currently failed to deploy the code on streamlit cloud due to streamlit cloud is run on linux, and on YOLOv5 have some path that define as window path. 


