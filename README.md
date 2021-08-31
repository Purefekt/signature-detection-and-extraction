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
- CNN based on VGG16 with bounding box regression layer -> A pre existing CNN architecture VGG16 was used. The final fully connected layer was removed and replaced with a bounding box regression layer. The dataset was the 776 images (subset of tobacco800) which had signatures. This model was used to predict the bounding box coordinates of signatures on documents.

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
```
To use conda environment in Jupyter Notebook, issue the following command
```
python -m ipykernel install --user --name=signature-detection-and-extraction
```
