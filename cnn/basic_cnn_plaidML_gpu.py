# to use plaidML as backend
# activate project environment
# plaidml-setup
# enable experimental device support: y
# use 1,2,3.. to select CPU, GPU etc
from os import environ
environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
from keras.preprocessing.image import ImageDataGenerator
import time
start_time = time.time()
print(keras.backend.backend())

train = ImageDataGenerator(rescale=1 / 255)
test = ImageDataGenerator(rescale=1 / 255)

train_dataset = train.flow_from_directory("/Users/veersingh/Desktop/tobacco800/cnn/train",
                                          target_size=(256, 256),
                                          batch_size=32,
                                          class_mode='binary',
                                          color_mode='grayscale')

cross_validation_dataset = test.flow_from_directory("/Users/veersingh/Desktop/tobacco800/cnn/cross_validation",
                                                    target_size=(256, 256),
                                                    batch_size=32,
                                                    class_mode='binary',
                                                    color_mode='grayscale')

print(cross_validation_dataset.class_indices)

# Model
model = keras.Sequential()

# Convolutional layer and maxpool layer 1
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
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

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# steps_per_epoch = train_imagesize/batch_size
model.fit_generator(train_dataset,
                    steps_per_epoch=27,
                    epochs=25,
                    validation_data=cross_validation_dataset
                    )

model.save('/Users/veersingh/Desktop/Internship/signature-detection-and-extraction/cnn/basic_cnn_model_gpu.h5')

print("--- %s seconds ---" % (time.time() - start_time))
