from os import environ

environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model
import numpy as np
import cv2
import os
import json
"""
This script uses the trained model and uses that to predict bbox coords for all images in the given directory.
In this case this dir is the evaluation dir which contains 155 images which the model has never seen before.
The xmin, ymin, xmax, ymax of each image will be saved in a json file.
Later we will use iou percentage to calculate how well this model worked.
"""

# load the trained model
model = load_model('bbox_regression_cnn.h5')

# create a list of paths of all images to be evaluated
eval_images_filenames_list = list()
eval_images_dir = '/Users/veersingh/Desktop/bbox_regression_dataset/eval'
for eval_image_filename in os.listdir(eval_images_dir):
    eval_images_filenames_list.append(eval_image_filename)

print(
    f'Number of images to be evaluated on --> {len(eval_images_filenames_list)}'
)

# initiate the predicted bbox dict which will be converted to json
eval_images_predicted_bbox_dict = dict()

# loop over the images that we'll be testing using our bounding box
# regression model
for eval_image_filename in eval_images_filenames_list:
    # load the input image (in Keras format) from disk and preprocess
    # it, scaling the pixel intensities to the range [0, 1]
    image = load_img(eval_images_dir + '/' + eval_image_filename,
                     target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # make bounding box predictions on the input image
    preds = model.predict(image)[0]
    (startX, startY, endX, endY) = preds

    # load the input image (in OpenCV format), resize it such that it
    # fits on our screen, and grab its dimensions
    image = cv2.imread(eval_images_dir + '/' + eval_image_filename)
    (h, w) = image.shape[:2]

    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)
    bbox = [startX, startY, endX, endY]

    # save this data to a dict
    eval_images_predicted_bbox_dict[eval_image_filename] = bbox

print(len(eval_images_predicted_bbox_dict))
print(eval_images_predicted_bbox_dict)

json_output = json.dumps(eval_images_predicted_bbox_dict, indent=4)
jsonFile = open('eval_images_predicted_bbox.json', 'w')
jsonFile.write(json_output)
jsonFile.close()
