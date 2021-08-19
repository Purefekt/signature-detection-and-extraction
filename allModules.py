import cv2
from method_1.module.signature_removal import Signature_removal
from method_2.module.loader import Loader
from method_2.module.extractor import Extractor
from method_2.module.boundingBox import BoundingBox
from method_4.module.core import extract_signature
import joblib
import numpy as np


class AllModules:

    def __init__(self, input_image_path):
        self.input_image_path = input_image_path

    def method_1(self):
        input_image_numpy_array = cv2.imread(self.input_image_path, cv2.IMREAD_GRAYSCALE)
        bbox_coords = Signature_removal(input_image_numpy_array).get_signature_bbox()
        return bbox_coords

    def method_2(self):
        loader = Loader()
        mask = loader.get_masks(self.input_image_path)[0]
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

        bbox_coords = [xmin, ymin, xmax, ymax]
        return bbox_coords

    def method_3(self):
        # use method 2
        loader = Loader()
        mask = loader.get_masks(self.input_image_path)[0]
        extractor = Extractor(amplfier=15)
        labeled_mask = extractor.extract(mask)
        try:
            xmin, ymin, w, h = BoundingBox().run(labeled_mask)
            xmax = xmin + w
            ymax = ymin + h
        # handle exception for when no bbox is found
        except:
            xmin, ymin, xmax, ymax = 0, 0, 0, 0

        # use method 1
        if (xmin and ymin and xmax and ymax) == 0:
            image = cv2.imread(self.input_image_path, cv2.IMREAD_GRAYSCALE)
            xmin, ymin, xmax, ymax = Signature_removal(image).get_signature_bbox()

        # Convert from numpy int64 to integer for JSON serialization
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

        bbox_coords = [xmin, ymin, xmax, ymax]
        return bbox_coords

    def method_4(self):
        # decision tree model
        model = joblib.load("method_4/module/models/decision-tree.pkl")
        clf = model

        im = cv2.imread(self.input_image_path, 0)
        mask = extract_signature(im, clf, preprocess=True)

        im = cv2.imread(self.input_image_path)
        im[np.where(mask == 255)] = (0, 0, 255)

        # find bounding box on image
        points = np.argwhere(mask == 255)
        points = np.fliplr(points)
        x, y, w, h = cv2.boundingRect(points)
        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h

        bbox_coords = [xmin, ymin, xmax, ymax]
        return bbox_coords

    def method_5(self):
        # apply method 4 first then method 1

        # decision tree model
        model = joblib.load("method_5/models/decision-tree.pkl")
        clf = model

        im = cv2.imread(self.input_image_path, 0)
        mask = extract_signature(im, clf, preprocess=True)

        im = cv2.imread(self.input_image_path)
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
            image = cv2.imread(self.input_image_path, cv2.IMREAD_GRAYSCALE)
            xmin, ymin, xmax, ymax = Signature_removal(image).get_signature_bbox()

        bbox_coords = [xmin, ymin, xmax, ymax]
        return bbox_coords

    def method_6(self):
        # apply method 4 then 2 then 1

        # decision tree model
        model = joblib.load("method_6/models/decision-tree.pkl")
        clf = model

        im = cv2.imread(self.input_image_path, 0)
        mask = extract_signature(im, clf, preprocess=True)

        im = cv2.imread(self.input_image_path)
        im[np.where(mask == 255)] = (0, 0, 255)

        # find bounding box on image
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
            mask = loader.get_masks(self.input_image_path)[0]
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
                image = cv2.imread(self.input_image_path, cv2.IMREAD_GRAYSCALE)
                xmin, ymin, xmax, ymax = Signature_removal(image).get_signature_bbox()

        bbox_coords = [xmin, ymin, xmax, ymax]
        return bbox_coords
