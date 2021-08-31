"""
to use plaidML as backend
activate project environment
plaidml-setup
enable experimental device support: y
use 1,2,3.. to select CPU, GPU etc"""
from os import environ

environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
from keras.preprocessing.image import ImageDataGenerator
import time
"""
This is a custom CNN which is trained to classify scanned documents as having a signature or not having one.
This model was trained on the entire Tobacco800 dataset (1290 images).
871 images in the train set (524 with signature, 347 without signature)
290 images in the cross validation set (174 with signature, 116 without signature)
129 images in the test set (78 with signature and 51 without signature)

This dataset can be found here -> https://www.kaggle.com/veersingh230799/custom-cnn-dataset-tobacco800
"""

# to see how long the script takes to run
start_time = time.time()

# Specify the paths to the train and cross validation datasets
train_dataset_path = '/Users/veersingh/Desktop/custom_cnn_dataset/train'
cross_validation_dataset_path = '/Users/veersingh/Desktop/custom_cnn_dataset/cross_validation'

train = ImageDataGenerator(rescale=1 / 255)
test = ImageDataGenerator(rescale=1 / 255)

train_dataset = train.flow_from_directory(directory=train_dataset_path,
                                          target_size=(256, 256),
                                          batch_size=32,
                                          class_mode='binary',
                                          color_mode='grayscale')

cross_validation_dataset = test.flow_from_directory(
    directory=cross_validation_dataset_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    color_mode='grayscale')

# print labels
print(
    f'Images without signature label -> {train_dataset.class_indices["no_signature"]}'
)
print(
    f'Images with signature label -> {train_dataset.class_indices["signature"]}'
)

# Model
model = keras.Sequential()

# Convolutional layer and maxpool layer 1
model.add(
    keras.layers.Conv2D(32, (3, 3),
                        activation='relu',
                        input_shape=(256, 256, 1)))
model.add(keras.layers.MaxPool2D(2, 2))

# Convolutional layer and maxpool layer 2
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D(2, 2))

# Convolutional layer and maxpool layer 3
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D(2, 2))

# Convolutional layer and maxpool layer 4
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(keras.layers.MaxPool2D(2, 2))

# This layer flattens the resulting image array to 1D array
model.add(keras.layers.Flatten())

# Hidden layer with 512 neurons and RelU activation function
model.add(keras.layers.Dense(512, activation='relu'))

# Output layer with single neuron which gives 0 for image w/o signature or 1 for image with signature
# Sigmoid activation function which makes our model output to lie between 0 and 1
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# print model details
print(model.summary())

# steps_per_epoch = train_imagesize/batch_size
model.fit_generator(train_dataset,
                    steps_per_epoch=27,
                    epochs=25,
                    validation_data=cross_validation_dataset)

# serialize the model
model.save('basic_cnn_model_gpu.h5')

# print how much time it took to run
print("--- %s seconds ---" % (time.time() - start_time))
