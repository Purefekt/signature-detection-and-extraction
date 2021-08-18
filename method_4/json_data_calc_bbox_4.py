import os
import json
import cv2
from method_4.module.core import extract_signature
import joblib
import numpy as np

images_dir = '/Users/veersingh/Desktop/docs_with_signs_dataset(tobacco800)/images'
json_file_output_dir = '/Users/veersingh/Desktop/docs_with_signs_dataset(tobacco800)'

calculated_values = dict()
model = joblib.load("module/models/decision-tree.pkl")
for filename in os.listdir(images_dir):
    current_image_name = filename
    current_image_path = images_dir + '/' + filename
    print(filename)

    # Use saifkhichi96/signature-extraction method
    clf = model
    im = cv2.imread(current_image_path, 0)
    mask = extract_signature(im, clf, preprocess=True)
    im = cv2.imread(current_image_path)
    im[np.where(mask == 255)] = (0, 0, 255)
    points = np.argwhere(mask == 255)
    points = np.fliplr(points)
    x, y, w, h = cv2.boundingRect(points)

    xmin = x
    ymin = y
    xmax = x + w
    ymax = y + h

    calculated_values[current_image_name] = [xmin, ymin, xmax, ymax]

# writing json output
json_output = json.dumps(calculated_values, indent=4)
output_json_file = json_file_output_dir + '/' + 'calculated_bbox_4.json'
jsonFile = open(output_json_file, "w")
jsonFile.write(json_output)
jsonFile.close()