import cv2
import numpy as np
from allModules import AllModules

img_path = '/Users/veersingh/Desktop/docs_with_signs_dataset(tobacco800)/images_less_than_10/bjn43c00-page02_1.tif'
img = cv2.imread(img_path, 0)
# img = cv2.bitwise_not(img)

kernel = np.ones((4, 4), np.uint8)
img_erosion = cv2.erode(img, kernel, iterations=1)
cv2.imwrite('/Users/veersingh/Desktop/Internship/signature-detection-and-extraction/images_with_0_iou/erosion.png', img_erosion)
# img_dilation = cv2.dilate(img, kernel, iterations=1)

input_image_path = '/Users/veersingh/Desktop/Internship/signature-detection-and-extraction/images_with_0_iou/erosion.png'

# model path for methods 4,5,6
model_path = '/Users/veersingh/Desktop/Internship/signature-detection-and-extraction/method_4/module/models/decision-tree.pkl'
method_1_bbox = AllModules(input_image_path=input_image_path).method_1()
method_2_bbox = AllModules(input_image_path=input_image_path).method_2()
method_3_bbox = AllModules(input_image_path=input_image_path).method_3()
method_4_bbox = AllModules(input_image_path=input_image_path).method_4(model_path=model_path)
# method_5_bbox = AllModules(input_image_path=input_image_path).method_5()
# method_6_bbox = AllModules(input_image_path=input_image_path).method_6()

print(method_1_bbox)
print(method_2_bbox)
print(method_3_bbox)
print(method_4_bbox)
# print(method_5_bbox)
# print(method_6_bbox)

cv2.imshow('Input', img)
cv2.imshow('Erosion', img_erosion)
# cv2.imshow('Dilation', img_dilation)

# cv2.waitKey(0)
