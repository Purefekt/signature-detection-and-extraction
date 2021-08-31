"""
to use plaidML as backend
activate project environment
plaidml-setup
enable experimental device support: y
use 1,2,3.. to select CPU, GPU etc"""
from os import environ

environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.applications import VGG16
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
"""
This is a custom CNN which is trained to classify scanned documents as having a signature or not having one.
This model was trained on a subset of the Tobacco800 dataset with all 776 images having a signature.
This dataset can be found here -> https://www.kaggle.com/veersingh230799/bbox-regression-dataset
"""

# to see how long the script takes to run
start_time = time.time()

# load the contents of the CSV annotations file
rows = open(
    '/Users/veersingh/Desktop/bbox_regression_dataset/ground_truth_bbox.csv'
).read().strip().split("\n")

data = []  # train images
targets = []  # bbox coords
filenames = []

# loop over the rows
for row in rows:
    # break the row into the filename and bounding box coordinates
    row = row.split(",")
    (filename, startX, startY, endX, endY) = row

    # derive the path to the input image, load the image (in OpenCV
    # format), and grab its dimensions
    imagePath = os.path.sep.join(
        ['/Users/veersingh/Desktop/bbox_regression_dataset/train', filename])
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    # scale the bounding box coordinates relative to the spatial
    # dimensions of the input image
    startX = float(startX) / w
    startY = float(startY) / h
    endX = float(endX) / w
    endY = float(endY) / h

    # load the image and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)

    # update our list of data, targets, and filenames
    data.append(image)
    targets.append((startX, startY, endX, endY))
    filenames.append(filename)

# partition data for training
# convert the data and targets to NumPy arrays, scaling the input
# pixel intensities from the range [0, 255] to [0, 1]
data = np.array(data, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")
# partition the data into training and testing splits using 90% of
# the data for training and the remaining 10% for testing
split = train_test_split(data,
                         targets,
                         filenames,
                         test_size=0.10,
                         random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]
# write the testing filenames to disk so that we can use then
# when evaluating/testing our bounding box regressor
f = open('test_images.txt', "w")
f.write("\n".join(testFilenames))
f.close()

# training the model
# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet",
            include_top=False,
            input_tensor=Input(shape=(224, 224, 3)))
# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False
# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)
# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)
# construct the model we will fine-tune for bounding box regression
model = Model(inputs=vgg.input, outputs=bboxHead)

# initialize the optimizer, compile the model, and show the model
# summary
model.compile(loss="mse", optimizer='adam')
print(model.summary())
# train the network for bounding box regression
print("training bounding box regressor...")
H = model.fit(trainImages,
              trainTargets,
              validation_data=(testImages, testTargets),
              batch_size=32,
              epochs=25,
              verbose=1)

# serialize the model to disk
print("[INFO] saving object detector model...")
model.save('bbox_regression_cnn.h5')
# plot the model training history
N = 25
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig('plot.png')

print("--- %s seconds ---" % (time.time() - start_time))
