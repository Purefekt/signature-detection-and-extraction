import json
from shutil import copyfile

"""
This script will create a dataset of images where the iou percentage was below 10% on all 6 methods.
These images will be investigated for why all methods failed on these images.
Further preprocessing will be applied on these images.
"""

# Read iou values for all 4 methods
iou_1_json = '/Users/veersingh/Desktop/docs_with_signs_dataset(tobacco800)/iou_1.json'
iou_2_json = '/Users/veersingh/Desktop/docs_with_signs_dataset(tobacco800)/iou_2.json'
iou_3_json = '/Users/veersingh/Desktop/docs_with_signs_dataset(tobacco800)/iou_3.json'
iou_4_json = '/Users/veersingh/Desktop/docs_with_signs_dataset(tobacco800)/iou_4.json'
iou_5_json = '/Users/veersingh/Desktop/docs_with_signs_dataset(tobacco800)/iou_5.json'
iou_6_json = '/Users/veersingh/Desktop/docs_with_signs_dataset(tobacco800)/iou_6.json'

with open(iou_1_json) as fhand:
    iou_1_data = json.load(fhand)
with open(iou_2_json) as fhand:
    iou_2_data = json.load(fhand)
with open(iou_3_json) as fhand:
    iou_3_data = json.load(fhand)
with open(iou_4_json) as fhand:
    iou_4_data = json.load(fhand)
with open(iou_5_json) as fhand:
    iou_5_data = json.load(fhand)
with open(iou_6_json) as fhand:
    iou_6_data = json.load(fhand)

# dataset with images
dataset_dir = '/Users/veersingh/Desktop/docs_with_signs_dataset(tobacco800)/images'
output_dir = '/Users/veersingh/Desktop/docs_with_signs_dataset(tobacco800)/images_less_than_10'

# Copy images which have less than 10% iou on all 4 methods
for filename in iou_1_data.keys():
    if iou_1_data[filename]['iou_in_percentage'] < 10:
        if iou_2_data[filename]['iou_in_percentage'] < 10:
            if iou_3_data[filename]['iou_in_percentage'] < 10:
                if iou_4_data[filename]['iou_in_percentage'] < 10:
                    if iou_5_data[filename]['iou_in_percentage'] < 10:
                        if iou_6_data[filename]['iou_in_percentage'] < 10:
                            print(filename)
                            # Copy this file to the new directory
                            source = dataset_dir + '/' + filename
                            destination = output_dir + '/' + filename
                            copyfile(source, destination)
