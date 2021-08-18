import cv2
from method_1.module.method_1_module import Signature_removal

test_image_path = '/Users/veersingh/Desktop/Internship/signature-detection-and-extraction/assets/aah97e00-page02_2.tif'

image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
xmin, ymin, xmax, ymax = Signature_removal(image).get_signature_bbox()

print(f'xmin -> {xmin}\n'
      f'ymin -> {ymin}\n'
      f'xmax -> {xmax}\n'
      f'ymax -> {ymax}')
