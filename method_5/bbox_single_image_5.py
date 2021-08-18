import cv2
from method_4.module.core import extract_signature
import joblib
import numpy as np
from method_1.module.method_1_module import Signature_removal

test_image_path = '/Users/veersingh/Desktop/Internship/signature-detection-and-extraction/assets/aah97e00-page02_2.tif'

# apply method 4 first then method 1

# decision tree model
model = joblib.load("models/decision-tree.pkl")
clf = model

im = cv2.imread(test_image_path, 0)
mask = extract_signature(im, clf, preprocess=True)

im = cv2.imread(test_image_path)
im[np.where(mask == 255)] = (0, 0, 255)

# find bounding box on image
points = np.argwhere(mask == 255)
points = np.fliplr(points)
x, y, w, h = cv2.boundingRect(points)
xmin = x
ymin = y
xmax = x + w
ymax = y + h

# If found coordinates are 0, use method 1
if (xmin and ymin and xmax and ymax) == 0:
    image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    xmin, ymin, xmax, ymax = Signature_removal(image).get_signature_bbox()

print(f'xmin -> {xmin}\n'
      f'ymin -> {ymin}\n'
      f'xmax -> {xmax}\n'
      f'ymax -> {ymax}')
