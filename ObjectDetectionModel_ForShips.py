


from ulrtalytics import YOLO

import matplotlib.pyplot as plt
import cv2
import squarify
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib.image as mpimg


##object detection project for ships


train_images = "DataSet/ships-aerial-images/train/images"
train_labels = "DataSet/ships-aerial-images/train/labels"

test_images = "DataSet/ships-aerial-images/test/images"
test_labels = "DataSet/ships-aerial-images/test/labels"


val_images = "DataSet/ships-aerial-images/valid/images"
val_labels = "DataSet/ships-aerial-images/valid/labels"






image_files = os.listdir()