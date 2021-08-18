import os
import json
import cv2
from method_4.module.core import extract_signature
import joblib
import numpy as np
from method_2.module.loader import Loader
from method_2.module.extractor import Extractor
from method_2.module.boundingBox import BoundingBox
from method_1.module.method_1_module import Signature_removal

images_dir = '/Users/veersingh/Desktop/docs_with_signs_dataset(tobacco800)/images'
json_file_output_dir = '/Users/veersingh/Desktop/docs_with_signs_dataset(tobacco800)'

calculated_values = dict()

# decision tree model
model = joblib.load("models/decision-tree.pkl")

for filename in os.listdir(images_dir):
    current_image_name = filename
    current_image_path = images_dir + '/' + filename
    print(filename)

    # method 4
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

    # If found coordinates are 0, use method 2
    if (xmin and ymin and xmax and ymax) == 0:
        loader = Loader()
        mask = loader.get_masks(current_image_path)[0]
        extractor = Extractor(amplfier=15)
        labeled_mask = extractor.extract(mask)
        try:
            xmin, ymin, w, h = BoundingBox().run(labeled_mask)
            xmax = xmin + w
            ymax = ymin + h
        # handle exception for when no bbox is found
        except:
            xmin, ymin, xmax, ymax = 0, 0, 0, 0

        # Convert from numpy int64 to integer for JSON serialization
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

        # # If found coordinates are 0, use method 1
        if (xmin and ymin and xmax and ymax) == 0:
            image = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
            xmin, ymin, xmax, ymax = Signature_removal(image).get_signature_bbox()

    calculated_values[current_image_name] = [xmin, ymin, xmax, ymax]

# writing json output
json_output = json.dumps(calculated_values, indent=4)
output_json_file = json_file_output_dir + '/' + 'calculated_bbox_6.json'
jsonFile = open(output_json_file, "w")
jsonFile.write(json_output)
jsonFile.close()
