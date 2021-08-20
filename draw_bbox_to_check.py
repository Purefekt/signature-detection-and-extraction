import cv2

img_path = '/Users/veersingh/Desktop/docs_with_signs_dataset(tobacco800)/images_less_than_10/bjn43c00-page02_1.tif'
img = cv2.imread(img_path)

xmin = 1026
ymin = 1220
xmax = 1201
ymax = 1283

cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 5)
cv2.imshow('Image', img)
cv2.waitKey(0)