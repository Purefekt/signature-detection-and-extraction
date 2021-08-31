from all_modules import AllModules
import json
import os
"""
This script will be used to predict the bounding boxes of signatures in documents using the 6 modules. This data will
be serialized into a JSON file containing all images in the following format:
{
    "filename.tif": [
        xmin,
        ymin,
        xmax,
        ymax
    ],
}
"""


def serialize_predicted_bbox(images_dir_path, module_number,
                             json_output_file_dir, json_output_filename):
    """
    Args:
        images_dir_path: full path of the directory containing test images
        module_number: The module which needs to be applied -> 1,2,3,4,5,6
        json_output_file_dir: The path of the output directory for the json file
        json_output_filename: the name of this output json file, eg 'output.json'

    Returns:
        JSON file
    """
    predicted_values = dict()
    for current_image_name in os.listdir(images_dir_path):
        current_image_path = images_dir_path + '/' + current_image_name

        # get bboxes
        if module_number == 1:
            bbox_coordinates = AllModules(
                input_image_path=current_image_path).module_1()
            predicted_values[current_image_name] = bbox_coordinates
        elif module_number == 2:
            bbox_coordinates = AllModules(
                input_image_path=current_image_path).module_2()
            predicted_values[current_image_name] = bbox_coordinates
        elif module_number == 3:
            bbox_coordinates = AllModules(
                input_image_path=current_image_path).module_3()
            predicted_values[current_image_name] = bbox_coordinates
        elif module_number == 4:
            bbox_coordinates = AllModules(
                input_image_path=current_image_path).module_4()
            predicted_values[current_image_name] = bbox_coordinates
        elif module_number == 5:
            bbox_coordinates = AllModules(
                input_image_path=current_image_path).module_5()
            predicted_values[current_image_name] = bbox_coordinates
        elif module_number == 6:
            bbox_coordinates = AllModules(
                input_image_path=current_image_path).module_6()
            predicted_values[current_image_name] = bbox_coordinates

    # writing json output
    json_output = json.dumps(predicted_values, indent=4)
    output_json_file = json_output_file_dir + '/' + json_output_filename
    jsonFile = open(output_json_file, "w")
    jsonFile.write(json_output)
    jsonFile.close()


# Getting predicted bbox for all images and serializing the data
images_dir_path = '/Users/veersingh/Desktop/docs_with_signs_dataset(tobacco800)/images'
json_output_file_dir = '/Users/veersingh/Desktop/json_output'

# For all 6 modules
serialize_predicted_bbox(images_dir_path=images_dir_path,
                         module_number=1,
                         json_output_file_dir=json_output_file_dir,
                         json_output_filename='predicted_bbox_1.json')
serialize_predicted_bbox(images_dir_path=images_dir_path,
                         module_number=2,
                         json_output_file_dir=json_output_file_dir,
                         json_output_filename='predicted_bbox_2.json')
serialize_predicted_bbox(images_dir_path=images_dir_path,
                         module_number=3,
                         json_output_file_dir=json_output_file_dir,
                         json_output_filename='predicted_bbox_3.json')
serialize_predicted_bbox(images_dir_path=images_dir_path,
                         module_number=4,
                         json_output_file_dir=json_output_file_dir,
                         json_output_filename='predicted_bbox_4.json')
serialize_predicted_bbox(images_dir_path=images_dir_path,
                         module_number=5,
                         json_output_file_dir=json_output_file_dir,
                         json_output_filename='predicted_bbox_5.json')
serialize_predicted_bbox(images_dir_path=images_dir_path,
                         module_number=6,
                         json_output_file_dir=json_output_file_dir,
                         json_output_filename='predicted_bbox_6.json')
