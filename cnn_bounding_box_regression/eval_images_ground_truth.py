import json
import os

"""
This script will create a json file for the ground truth bbox coordinates for the images in the evaluation dataset
"""

# get filenames for all images in the test directory
eval_images_dir = '/Users/veersingh/Desktop/bbox_regression_dataset/eval'
eval_images = os.listdir(eval_images_dir)
eval_images_filenames_list = list()
for eval_image_filename in eval_images:
    eval_images_filenames_list.append(eval_image_filename)

# initialize dict for converting to json
eval_images_ground_truth = dict()

# read the ground truth json file for all images
ground_truth_bbox_json = '/Users/veersingh/Desktop/bbox_regression_dataset/ground_truth_bbox.json'
f = open(ground_truth_bbox_json, )
ground_truth_bbox_data = json.load(f)
f.close()

for filename in ground_truth_bbox_data.keys():
    if filename in eval_images_filenames_list:
        eval_images_ground_truth[filename] = ground_truth_bbox_data[filename]

json_output = json.dumps(eval_images_ground_truth, indent=4)
jsonFile = open('eval_images_ground_truth_bbox.json', 'w')
jsonFile.write(json_output)
jsonFile.close()
