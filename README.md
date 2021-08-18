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
