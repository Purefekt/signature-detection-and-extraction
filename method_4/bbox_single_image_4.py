import cv2
from method_4.module.core import extract_signature
import joblib
import numpy as np


input_image = '/Users/veersingh/Desktop/signature-detection-and-extraction/assets/aah97e00-page02_2.tif'

# decision tree model
model = joblib.load("module/models/decision-tree.pkl")
clf = model

im = cv2.imread(input_image, 0)
mask = extract_signature(im, clf, preprocess=True)

im = cv2.imread(input_image)
im[np.where(mask==255)] = (0, 0, 255)

# find bounding box on image
points = np.argwhere(mask==255)  # find where the black pixels are
points = np.fliplr(points)       # store them in x,y coordinates instead of row,col indices
x, y, w, h = cv2.boundingRect(points)  # create a rectangle around those points
xmin = x
ymin = y
xmax = x + w
ymax = y + h

print(f'xmin -> {xmin}\n'
      f'ymin -> {ymin}\n'
      f'xmax -> {xmax}\n'
      f'ymax -> {ymax}')
