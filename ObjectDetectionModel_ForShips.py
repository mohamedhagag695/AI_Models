


from ultralytics import YOLO



import matplotlib.pyplot as plt
import cv2
import squarify
import os
import torch
import random
import pandas as pd
import seaborn as sns
import matplotlib.image as mpimg


##object detection project for ships

print(torch.cuda.is_available())


train_images = "DataSet/ships-aerial-images/train/images"
train_labels = "DataSet/ships-aerial-images/train/labels"

test_images = "DataSet/ships-aerial-images/test/images"
test_labels = "DataSet/ships-aerial-images/test/labels"


val_images = "DataSet/ships-aerial-images/valid/images"
val_labels = "DataSet/ships-aerial-images/valid/labels"




# image_files = os.listdir(train_images)
# print(len(image_files))



# # Get a list of all the image files in the training images directory
# image_files = os.listdir(train_images)

# # Choose 16 random image files from the list
# random_images = random.sample(image_files, 16)

# # Set up the plot
# fig, axs = plt.subplots(4, 4, figsize=(16, 16))

# # Loop over the random images and plot the object detections
# for i, image_file in enumerate(random_images):
#     row = i // 4
#     col = i % 4
    
#     # Load the image
#     image_path = os.path.join(train_images, image_file)
#     image = cv2.imread(image_path)

#     # Load the labels for this image
#     label_file = os.path.splitext(image_file)[0] + ".txt"

#     label_path = os.path.join(train_labels, label_file)
#     with open(label_path, "r") as f:
#         labels = f.read().strip().split("\n")

#     # Loop over the labels and plot the object detections
#     # Loop over the labels and plot the object detections
#     for label in labels:
#         if len(label.split()) != 5:
#             continue
#         class_id, x_center, y_center, width, height = map(float, label.split())
#         x_min = int((x_center - width/2) * image.shape[1])
#         y_min = int((y_center - height/2) * image.shape[0])
#         x_max = int((x_center + width/2) * image.shape[1])
#         y_max = int((y_center + height/2) * image.shape[0])
#         cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


#     # Show the image with the object detections
#     axs[row, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     axs[row, col].axis('off')

# plt.show()

# # Load an image using OpenCV

# image = cv2.imread("DataSet/ships-aerial-images/test/images/02e39612d_jpg.rf.cc5483bb711f080d12b644ff62cf977a.jpg")

# # Get the size of the image
# height, width, channels = image.shape
# print(f"The image has dimensions {width}x{height} and {channels} channels.")

# model = YOLO('yolov8x.pt')

# # Training the model
# model.train(data = 'DataSet/ships-aerial-images/data.yaml',
#             epochs = 20,
#             imgsz = height,
#             seed = 42,
#             batch = 8, workers = 4)