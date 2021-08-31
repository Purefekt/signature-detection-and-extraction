# path to training dataset_docs_with_signs
train_images_path = '/Users/veersingh/Desktop/bbox_regression_dataset/train'

# path to bbox csv
ground_truth_bbox_path = '/Users/veersingh/Desktop/bbox_regression_dataset/ground_truth_bbox.csv'

# path to trained model
model_path = 'bbox_regression_cnn.h5'

# path to evaluation dataset_docs_with_signs
eval_images_path = '/Users/veersingh/Desktop/bbox_regression_dataset/eval'

# test file names
test_filenames_path = 'test_images.txt'

# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 32

# plot path
plot_path = 'plot.png'
