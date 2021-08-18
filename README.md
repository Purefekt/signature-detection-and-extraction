# Signature Detection and Extraction

## Install with conda
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

conda install ipykernel

conda install -c anaconda pandas
```
To use conda environment in Jupyter Notebook, issue the following command
```
python -m ipykernel install --user --name=signature-detection-and-extraction
```

## Resources
- Method 1 --> Signature Extractor - Adapted from https://github.com/ahmetozlu/signature_extractor
- Method 2 --> Signature Detect - https://pypi.org/project/signature-detect/
- Method 3 --> Hybrid between Method 1 and 2
- Method 4 --> https://github.com/saifkhichi96/signature-extraction