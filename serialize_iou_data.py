import json
"""
This script uses the serialized ground truth bbox data, the predicted bbox data and calculates the iou percentage.
Then it saves that information in a JSON file in the format:
{
    "filename.tif": {
        "ground_truth": [
            xmin,
            ymin,
            xmax,
            ymax
        ],
        "calculated": [
            xmin,
            ymin,
            xmax,
            ymax
        ],
        "intersection": [
            xmin,
            ymin,
            xmax,
            ymax
        ],
        "iou_in_percentage": 0.0
    },    
}
"""


def get_evaluation_json(ground_truth_values_json, predicted_values_json,
                        output_json_name):
    """
    Args:
        ground_truth_values_json: path to the json file containing ground truth bbox values
        predicted_values_json: path to the json file containing predicted bbox values
        output_json_name: name of the output json file

    Returns:
        json file containing iou percentages in the given format above ^

    """
    f = open(ground_truth_values_json,)
    ground_truth_values = json.load(f)
    f.close()

    f = open(predicted_values_json,)
    calculated_values = json.load(f)
    f.close()

    iou_dict = dict()
    for filename in ground_truth_values.keys():
        # load ground truth and calculated coordinate values
        xmin_gt = ground_truth_values[filename][0]
        ymin_gt = ground_truth_values[filename][1]
        xmax_gt = ground_truth_values[filename][2]
        ymax_gt = ground_truth_values[filename][3]

        xmin_c = calculated_values[filename][0]
        ymin_c = calculated_values[filename][1]
        xmax_c = calculated_values[filename][2]
        ymax_c = calculated_values[filename][3]

        # Calculate iou
        # ground truth and predicted bbox areas
        gt_bbox_area = (xmax_gt - xmin_gt) * (ymax_gt - ymin_gt)
        calc_bbox_area = (xmax_c - xmin_c) * (ymax_c - ymin_c)

        # Coordinates of intersection
        xmin_inter = max(xmin_gt, xmin_c)
        ymin_inter = max(ymin_gt, ymin_c)
        xmax_inter = min(xmax_gt, xmax_c)
        ymax_inter = min(ymax_gt, ymax_c)

        if xmax_inter < xmin_inter or ymax_inter < ymin_inter:
            iou = 0.0
        else:
            intersection_area = (xmax_inter - xmin_inter) * (ymax_inter -
                                                             ymin_inter)
            iou = intersection_area / float(gt_bbox_area + calc_bbox_area -
                                            intersection_area)
            iou = round(iou * 100, 2)

        # Print filename and iou %age
        print(f'{filename} --> {iou}% accurate')

        # add to dict
        iou_dict.update({
            filename: {
                "ground_truth": [xmin_gt, ymin_gt, xmax_gt, ymax_gt],
                "calculated": [xmin_c, ymin_c, xmax_c, ymax_c],
                "intersection": [
                    xmin_inter, ymin_inter, xmax_inter, ymax_inter
                ],
                "iou_in_percentage": iou
            }
        })

    # writing json output
    json_output = json.dumps(iou_dict, indent=4)
    output_json_file = output_json_name
    jsonFile = open(output_json_file, "w")
    jsonFile.write(json_output)
    jsonFile.close()


ground_truth_values_json = 'assets/ground_truth_bbox.json'

predicted_values_module_1_json = 'assets/predicted_bbox_1.json'
output_json_module_1_name = 'assets/iou_1.json'

predicted_values_module_2_json = 'assets/predicted_bbox_2.json'
output_json_module_2_name = 'assets/iou_2.json'

predicted_values_module_3_json = 'assets/predicted_bbox_3.json'
output_json_module_3_name = 'assets/iou_3.json'

predicted_values_module_4_json = 'assets/predicted_bbox_4.json'
output_json_module_4_name = 'assets/iou_4.json'

predicted_values_module_5_json = 'assets/predicted_bbox_5.json'
output_json_module_5_name = 'assets/iou_5.json'

predicted_values_module_6_json = 'assets/predicted_bbox_6.json'
output_json_module_6_name = 'assets/iou_6.json'

get_evaluation_json(ground_truth_values_json=ground_truth_values_json,
                    predicted_values_json=predicted_values_module_1_json,
                    output_json_name=output_json_module_1_name)

get_evaluation_json(ground_truth_values_json=ground_truth_values_json,
                    predicted_values_json=predicted_values_module_2_json,
                    output_json_name=output_json_module_2_name)

get_evaluation_json(ground_truth_values_json=ground_truth_values_json,
                    predicted_values_json=predicted_values_module_3_json,
                    output_json_name=output_json_module_3_name)

get_evaluation_json(ground_truth_values_json=ground_truth_values_json,
                    predicted_values_json=predicted_values_module_4_json,
                    output_json_name=output_json_module_4_name)

get_evaluation_json(ground_truth_values_json=ground_truth_values_json,
                    predicted_values_json=predicted_values_module_5_json,
                    output_json_name=output_json_module_5_name)

get_evaluation_json(ground_truth_values_json=ground_truth_values_json,
                    predicted_values_json=predicted_values_module_6_json,
                    output_json_name=output_json_module_6_name)
