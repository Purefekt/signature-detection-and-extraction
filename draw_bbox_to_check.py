import cv2

img_path = '/Users/veersingh/Desktop/docs_with_signs_dataset(tobacco800)/images/yrz52d00.tif'
img = cv2.imread(img_path)

xmin = 894
ymin = 1594
xmax = 1295
ymax = 1718

cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 5)
cv2.imshow('Image', img)
cv2.waitKey(0)
