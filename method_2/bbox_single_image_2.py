from method_2.module.loader import Loader
from method_2.module.extractor import Extractor
from method_2.module.boundingBox import BoundingBox

path = '/Users/veersingh/Desktop/signature-detection-and-extraction/assets/aah97e00-page02_2.tif'

loader = Loader()
mask = loader.get_masks(path)[0]
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

print(f'xmin -> {xmin}\n'
      f'ymin -> {ymin}\n'
      f'xmax -> {xmax}\n'
      f'ymax -> {ymax}')
