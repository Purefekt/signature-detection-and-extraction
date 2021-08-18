import cv2
from method_1.module.method_1_module import Signature_removal
from method_2.module.loader import Loader
from method_2.module.extractor import Extractor
from method_2.module.boundingBox import BoundingBox

# First calculate bbox using method 2, if all coordinates are 0 then use method 1

image_path = '/Users/veersingh/Desktop/Internship/signature-detection-and-extraction/assets/aah97e00-page02_2.tif'

# use method 2
loader = Loader()
mask = loader.get_masks(image_path)[0]
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
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    xmin, ymin, xmax, ymax = Signature_removal(image).get_signature_bbox()

# Convert from numpy int64 to integer for JSON serialization
xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

print(f'xmin -> {xmin}\n'
      f'ymin -> {ymin}\n'
      f'xmax -> {xmax}\n'
      f'ymax -> {ymax}')