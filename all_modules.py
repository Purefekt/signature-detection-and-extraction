import cv2
from modules.module_1.signature_removal import Signature_removal
from modules.module_2.loader import Loader
from modules.module_2.extractor import Extractor
from modules.module_2.boundingBox import BoundingBox
from modules.module_4.core import extract_signature
import joblib
import numpy as np


class AllModules:
    """
    This class combines all 6 modules in one. There are 6 methods corresponding to the 6 modules

    Attributes:
        input_image_path: path of the input image
    """

    def __init__(self, input_image_path):
        self.input_image_path = input_image_path

    def module_1(self):
        """
        Args:
            self

        Returns:
            bbox coords of the signature calculated using module 1
        """
        input_image_numpy_array = cv2.imread(self.input_image_path,
                                             cv2.IMREAD_GRAYSCALE)
        bbox_coords = Signature_removal(
            input_image_numpy_array).get_signature_bbox()
        return bbox_coords

    def module_2(self):
        """
        Args:
            self

        Returns:
            bbox coords of the signature calculated using module 2
        """
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

    def module_3(self):
        """
        Args:
            self

        Returns:
            bbox coords of the signature calculated using module 3
        """
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
            xmin, ymin, xmax, ymax = Signature_removal(
                image).get_signature_bbox()

        # Convert from numpy int64 to integer for JSON serialization
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

        bbox_coords = [xmin, ymin, xmax, ymax]
        return bbox_coords

    def module_4(self, model_path="modules/model_4_5_6/decision-tree.pkl"):
        """
        Args:
            self
            model_path: path to the pre trained decision tree classifier model

        Returns:
            bbox coords of the signature calculated using module 4
        """
        # decision tree model
        model = joblib.load(model_path)
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

    def module_5(self, model_path="modules/model_4_5_6/decision-tree.pkl"):
        """
        Args:
            self
            model_path: path to the pre trained decision tree classifier model

        Returns:
            bbox coords of the signature calculated using module 5
        """
        # apply method 4 first then method 1

        # decision tree model
        model = joblib.load(model_path)
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
            xmin, ymin, xmax, ymax = Signature_removal(
                image).get_signature_bbox()

        bbox_coords = [xmin, ymin, xmax, ymax]
        return bbox_coords

    def module_6(self, model_path="modules/model_4_5_6/decision-tree.pkl"):
        """
        Args:
            self
            model_path: path to the pre trained decision tree classifier model

        Returns:
            bbox coords of the signature calculated using module 6
        """
        # apply method 4 then 2 then 1

        # decision tree model
        model = joblib.load(model_path)
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
                xmin, ymin, xmax, ymax = Signature_removal(
                    image).get_signature_bbox()

        bbox_coords = [xmin, ymin, xmax, ymax]
        return bbox_coords
