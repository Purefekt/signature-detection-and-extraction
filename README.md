# Signature Detection and Extraction

## Summary
This repository explore signature detection and extraction using image processing methods and convolutional neural networks.
- For image processing methods we have 6 modules (3 open source modules and 3 hybrids of those open source modules)
- For CNNs we have a custom CNN to detect if a given document has a signature or not and a CNN with bounding box regression to get the bounding box coordinates for a given document.

## Image Processing modules
- Module 1 -> Adapted from https://github.com/ahmetozlu/signature_extractor. This project aims at removing the signatures from a given document. I have modified it to extract the signature instead and then used morphological techniques to find the bounding box for the signature.
- Module 2 -> Adapted from https://github.com/EnzoSeason/signature_detection. This project extracts the signature in a given document, but it didnt output the bounding box coordinates. I modified this to instead output the bounding box coordinates of the signature.
- Module 3 -> This is a hybrid of modules 1 and 2.
- Module 4 -> Adapted from https://github.com/saifkhichi96/signature-extraction. This project has a gui, it lets the user pick an image and then extracts the signature using a decision tree classifier which was trained a dataset of 8000 signatures. I modified it by removing the gui since it was really slow while testing on 700+ images and also made it output just the bounding box coordinates for the signature.
- Module 5 -> Hybrid of modules 1 and 4.
- Module 6 -> Hybrid of modules 1,2 and 4.

## Convolutional Neural Network
- Custom CNN classifier -> A custom CNN was built using Keras. This CNN was trained on the entire Tobacco800 dataset (1290 images). Images with a signature (776 images) were labelled as 1 and images without a signature were labelled as 0 (514).
    - Download the trained model here -> https://drive.google.com/file/d/1Omh0ltYQvBJJo5h-kO5o2CNQjLEPqJSF/view?usp=sharing
- CNN based on VGG16 with bounding box regression layer -> A pre existing CNN architecture VGG16 was used. The final fully connected layer was removed and replaced with a bounding box regression layer. The dataset was the 776 images (subset of tobacco800) which had signatures. This model was used to predict the bounding box coordinates of signatures on documents.
    - Download the trained model here -> https://drive.google.com/file/d/14nq_srPz3qKDU0l_8jN3Ixa7Q01gITP1/view?usp=sharing

## Evaluating Different Resources
This Jupyter notebook contains the evaluation - [evaluating_different_signature_detection_methods.ipynb](evaluating_different_signature_detection_methods.ipynb)

## Install prerequisites
ImageMagick (for macOS use brew)
```
brew install imagemagick
```
## Install python packages with conda
**Note** --> Install all packages one by one, requirements.txt has some issues because of different channels and opencv-contrib needs to be built from source.

Create conda environment with the name **signature-detection-and-extraction** and activate virtual environment
```
conda create --name signature-detection-and-extraction

conda activate signature-detection-and-extraction
```
Install packages one by one
```
conda install -c anaconda joblib

conda install -c anaconda numpy

conda install -c anaconda scikit-learn

conda install -c anaconda scipy

conda install -c conda-forge opencv

conda install -c conda-forge threadpoolctl

conda install -c conda-forge tqdm
```
opencv-contrib package will have to be built from source with NONFREE flag set. If this package is installed directly then we wont be able to use a crucial function in the code. This might take some time.
```
CMAKE_ARGS="-DOPENCV_ENABLE_NONFREE=ON" pip install --no-binary=opencv-contrib-python opencv-contrib-python

```
```
conda install -c anaconda scikit-image

conda install -c conda-forge notebook

conda install -c anaconda pandas

conda install -c conda-forge notebook

conda install -c anaconda keras

pip install -U plaidml-keras
```
To use conda environment in Jupyter Notebook, issue the following command
```
python -m ipykernel install --user --name=signature-detection-and-extraction
```
## Directory tree
```
Project root/
├── 6_modules_examples.ipynb
├── all_modules.py
├── assets/
│   ├── ...
├── cnn_bounding_box_regression/
│   ├── bbox_regression_cnn.h5 (download seperately)
│   ├── cnn_bbox_regression.py
│   ├── create_ground_truth_csv.py
│   ├── eval_images_ground_truth.py
│   ├── eval_images_ground_truth_bbox.json
│   ├── eval_images_predicted_bbox.json
│   ├── evaluating_bbox_regression_cnn.ipynb
│   ├── iou_bbox_regression.json
│   ├── plot.png
│   ├── predictions_cnn_bbox_regression.py
│   └── test_images.txt
├── cnn_class_label_predictor/
│   ├── basic_cnn_model_gpu.h5 (download seperately)
│   ├── basic_cnn_plaidML_gpu.py
│   ├── cleaning_data.py
│   ├── cpu_vs_gpu.png
│   ├── evaluating_basic_cnn.ipynb
│   └── test/
│       ├── ...
├── dataset_docs_with_signs/
│   └── cleaning_data.py
├── evaluating_different_signature_extraction_modules.ipynb
├── modules/
│   ├── model_4_5_6/
│   │   └── decision-tree.pkl
│   ├── module_1/
│   │   ├── ...
│   ├── module_2/
│   │   ├── ...
│   └── module_4/
│       ├── ...
├── README.md
├── serialize_ground_truth_bbox.py
├── serialize_iou_data.py
└── serialize_predicted_bbox.py
```

## Summary of all files
- ```6_modules_examples.ipynb``` -> jupyter notebook with examples on how to use all 6 modules to get bbox coordinates of signature in an image
- ```all_modules.py``` -> class containing methods for implementing all 6 signature extraction modules
- ```assets/``` -> directory containing resources like test image, json file with bbox coord data, etc
- ```cnn_bounding_box_regression/bbox_regression_cnn.h5``` -> trained model CNN for bbox regression
- ```cnn_bounding_box_regression/cnn_bbox_regression.py``` -> python script to train the CNN model for bbox regression
- ```cnn_bounding_box_regression/create_ground_truth_csv.py``` -> python script to convert ground truth bbox values into a csv file for input for the CNN
- ```cnn_bounding_box_regression/eval_images_ground_truth.py``` -> python scrip for getting the ground truth bbox values for evaluation set
- ```cnn_bounding_box_regression/eval_images_ground_truth_bbox.json``` -> json file containing the ground truth bbox values for images in evaluation set
- ```cnn_bounding_box_regression/eval_images_predicted_bbox.json``` -> json file containing the predicted bbox values for the images in the evaluation dataset
- ```cnn_bounding_box_regression/evaluating_bbox_regression_cnn.ipynb``` -> jupyter notebook showing results for the bbox regression CNN model
- ```cnn_bounding_box_regression/iou_bbox_regression.json``` -> json file containing the iou values for the images in the evaluation dataset
- ```cnn_bounding_box_regression/plot.png``` -> loss over 25 epochs while training the model
- ```cnn_bounding_box_regression/predictions_cnn_bbox_regression.py``` -> python script which predicts the bboxes on the entire eval dataset using the trained model
- ```cnn_bounding_box_regression/test_images.txt``` -> list of filenames of images in the test set
- ```cnn_class_label_predictor/basic_cnn_model_gpu.h5``` -> trained custom CNN for classification
- ```cnn_class_label_predictor/basic_cnn_plaidML_gpu.py``` -> python script to train the custom CNN with plaidML backend to use AMG GPU
- ```cnn_class_label_predictor/cleaning_data.py``` -> python script to create a custom dataset
- ```cnn_class_label_predictor/cpu_vs_gpu.png``` -> image showing the difference in time while using cpu and gpu for training a model
- ```cnn_class_label_predictor/evaluating_basic_cnn.ipynb``` -> jupyter notebook showing results for the basic custom CNN model
- ```cnn_class_label_predictor/test/``` -> directory containing the test files. These are used by the ```cnn_class_label_predictor/evaluating_basic_cnn.ipynb``` jupyter notebook
- ```dataset_docs_with_signs/cleaning_data.py``` -> python script to create custom dataset
- ```evaluating_different_signature_extraction_modules.ipynb``` -> results for the 6 modules used for signature extraction
- ```modules/model_4_5_6/decision-tree.pkl``` -> trained decision tree model used by modules 4,5,6
- ```modules/module_1/``` -> all files used by module 1
- ```modules/module_2/``` -> all files used by module 2
- ```modules/module_4/``` -> all files used by module 3
- ```serialize_ground_truth_bbox.py``` -> python script to convert xml ground truth bbox data files into one json file
- ```serialize_iou_data.py``` -> python script which uses the ground truth bbox json file and predicted bbox json file to calculate iou and serializes that information into a new json file
- ```serialize_predicted_bbox.py``` -> python script which uses all modules and predicts bboxes for all images and serializes that data into seperate json files for each module