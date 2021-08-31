import os
import shutil
"""
Dataset Used: Tobacco800 -> https://www.kaggle.com/veersingh230799/tobacco800-dataset
This dataset_docs_with_signs has 1290 images of scanned documents. Some have signatures and some dont. This script uses 
the xml ground truth files to identify if an image has a signature. If the image has a signature then it is put into the 
images_with_sig directory otherwise it is put into the images_without_sig directory.
"""

ground_truth_dir = '/Users/veersingh/Desktop/tobacco800/ground_truth_xml'
all_images_dir = '/Users/veersingh/Desktop/tobacco800/all_images'
images_with_sig_dir = '/Users/veersingh/Desktop/tobacco800/images_with_sig'
images_without_sig_dir = '/Users/veersingh/Desktop/tobacco800/images_without_sig'

total = 0
with_signature = 0
without_signature = 0
for filename in os.listdir(ground_truth_dir):
    total = total + 1
    # image file of the ground truth xml file
    corresponding_image_file_name = filename.replace('.xml', '.tif')
    corresponding_image_file_path = all_images_dir + '/' + corresponding_image_file_name

    current_file_path = ground_truth_dir + '/' + filename
    fhand = open(current_file_path, 'r')
    read_file = fhand.read()
    # if file contains this string, then the image has a signature
    # copy this file to the images_with_sig dir, otherwise copy it to images_without_sig dir
    check_string = 'DLSignature'
    if check_string in read_file:
        with_signature = with_signature + 1
        print(corresponding_image_file_path)
        shutil.copyfile(
            corresponding_image_file_path,
            images_with_sig_dir + '/' + corresponding_image_file_name)
    else:
        without_signature = without_signature + 1
        print(corresponding_image_file_path)
        shutil.copyfile(
            corresponding_image_file_path,
            images_without_sig_dir + '/' + corresponding_image_file_name)
    fhand.close()

print(total)
print(with_signature)
print(without_signature)
