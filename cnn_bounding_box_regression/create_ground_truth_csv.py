import json
import pandas as pd
import os
"""
This script converts the ground truth bbox values into a csv file which will be used in the bounding box regression
CNN model. Once the csv file is ready make sure to delete row 1 and column A since these are the automatic index. 
"""

# get names of all images in the train set (621 images)
train_images_dir = '/Users/veersingh/Desktop/bbox_regression_dataset/train'
train_images = os.listdir(train_images_dir)
train_images_filenames_list = list(())
for train_image_filename in train_images:
    train_images_filenames_list.append(train_image_filename)

ground_truth_bbox_json = '/Users/veersingh/Desktop/bbox_regression_dataset/ground_truth_bbox.json'
f = open(ground_truth_bbox_json,)
ground_truth_bbox_data = json.load(f)
f.close()

col_0 = list()  # filenames
col_1 = list()  # xmin
col_2 = list()  # ymin
col_3 = list()  # xmax
col_4 = list()  # ymax

for filename in ground_truth_bbox_data.keys():
    if filename in train_images_filenames_list:
        col_0.append(filename)
        col_1.append(ground_truth_bbox_data[filename][0])
        col_2.append(ground_truth_bbox_data[filename][1])
        col_3.append(ground_truth_bbox_data[filename][2])
        col_4.append(ground_truth_bbox_data[filename][3])

df = pd.DataFrame(list(zip(col_0, col_1, col_2, col_3, col_4)))
df.to_csv('ground_truth_bbox.csv')
