# A python script to preprocess images from the CBIS-DDSM
# Based on processed dataframes created using DDSM_Processing
import argparse
import pandas as pd
from ast import literal_eval
import pydicom as dicom
import cv2
import os
import numpy as np


def parse_options():
    """
    Parse the arguments from the user input in the command line which contains filepaths
    :return: arguments from user input
    """
    parser = argparse.ArgumentParser(description='Reformat CBIS-DDSM Dataset')
    parser.add_argument('--data', required=True, help='folder where processed data is read from and saved to')
    parser.add_argument('--images', required=True, help='folder where images will be saved')
    args = parser.parse_args()
    return args


def read_dataset_information(path):
    """
    reads a csv file with relevant scan information
    :param path: (String) file path of csv file
    :return: (Pandas Dataframe) dataframe used to sort data
    """
    df = pd.read_csv(path, index_col=None)
    df.Class = df.Class.apply(literal_eval)                     # Column contains list, not string
    df.Mask_Location = df.Mask_Location.apply(literal_eval)     # Column contains, not string
    df['Box_Coords'] = ""                                       # New column needed to store localisation points
    df['Filename'] = ""                                         # New column to store saved filename
    df['Image_Dimensions'] = ""                                 # New column to store image size
    return df


def read_dicom(filepath, scan_type='roi'):
    """
    reads an image in the dicom format
    :param filepath: (String) the filepath of the image to load
    :param scan_type: (String) type of imaging being loaded
    :return: (Numpy array) image in numpy array format
    """
    image = dicom.dcmread(filepath)
    image = image.pixel_array                                   # Dicom images contain the pixels of the image
    if scan_type == 'mammogram':                                # Only the full mammograms need converting to unit8
        image = (image / 256).astype('uint8')
    return image


def get_tumour_mask_paths(scan_object):
    """
    gets the locations of the ground truth files
    :param scan_object: (Tuple object) scan being processed in batch
    :return: (List) list of directories where ground truth files are saved
    """
    # Create list of paths to tumour masks
    tumour_mask_dirs = []
    # Process each current filepath for ROI
    for directory in scan_object.Mask_Location:
        # List all files at the end of a path
        list_of_files = filter(lambda x: os.path.isfile(os.path.join(directory, x)), os.listdir(directory))
        # Sort files by decreasing filesize
        list_of_files = sorted(list_of_files, key=lambda x: os.stat(os.path.join(directory, x)).st_size, reverse=True)
        # Get first filepath and add to list
        tumour_mask_file = list_of_files[0]
        tumour_mask_dirs.append(os.path.join(directory, tumour_mask_file))
    return tumour_mask_dirs


def get_bounding_boxes(paths):
    """
    gets box coordinates for ground truth masks
    :param paths: (List) list of ground truth directories
    :return: (List) list of bounding box coordinates
    """
    bboxes = []
    for path in paths:
        tumour_mask = read_dicom(path)
        # store box coordinates as xmin, ymin, width and height
        x, y, w, h = cv2.boundingRect(tumour_mask)
        box_coords = [x, y, w, h]
        bboxes.append(box_coords)
    return bboxes


def adjust_bounding_boxes(box_list, x_change, y_change):
    """
    ensures the position of the ROI is altered with preprocessing
    :param box_list: (List) list of ROI coordinates
    :param x_change: (Integer) value to change x position by
    :param y_change: (Integer) value to change y position by
    :return:
    """
    for box in box_list:
        box[0] = box[0] - x_change
        box[1] = box[1] - y_change
    return box_list


def save_mammogram(img, folder, file):
    """
    saves the mammogram image to a specific filepath
    :param img: (numpy array) mammogram to be saved
    :param folder: (String) folder dpath
    :param file: (String) file name
    :return:
    """
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, file), img)


def valid_box(value):
    """
    checks if the box ROI coordinates are valid
    :param value: (Integer) coordinate to check
    :return: (Integer) checked coordinate
    """
    if value < 0:
        value = 0
        return value
    else:
        return value


def preprocessing(df, target):
    """
    main preprocessing function
    :param df: (Pandas Dataframe) dataframe containing mammogram information
    :param target: (String) target folder to save images
    :return: (Pandas Dataframe) dataframe containing preprocessed mammogram information
    """
    # Process every full mammogram by looping over dataframe
    for scan in df.itertuples():

        print(f"Processing Mammogram {scan[0] + 1} / {len(df.index) + 1}...")
        # Update the dataframe with precise file paths for the tumour maths
        tumour_mask_paths = get_tumour_mask_paths(scan)
        df.at[scan[0], 'Mask_Location'] = tumour_mask_paths

        # Load mammogram
        mammogram = read_dicom(scan.Full_Mammogram_Location, 'mammogram')

        # Get bounding box coordinates for tumour
        bounding_boxes = get_bounding_boxes(tumour_mask_paths)

        # Crop borders
        row_count, column_count = mammogram.shape
        x1_crop = int(column_count * 0.01)
        x2_crop = int(column_count * (1 - 0.01))
        y1_crop = int(row_count * 0.04)
        y2_crop = int(row_count * (1 - 0.04))
        mammogram = mammogram[y1_crop:y2_crop, x1_crop:x2_crop]

        # Adjust borders
        bounding_boxes = adjust_bounding_boxes(bounding_boxes, x1_crop, y1_crop)

        # Remove noise
        blur = cv2.GaussianBlur(mammogram, (5, 5), 0)

        # Binary threshold on the gray scan
        _, mammogram_threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(23, 23))
        mammogram_threshold = cv2.morphologyEx(mammogram_threshold, cv2.MORPH_OPEN, kernel)
        mammogram_threshold = cv2.morphologyEx(mammogram_threshold, cv2.MORPH_DILATE, kernel)

        # Find the largest contour in threshold mammogram
        all_contours, _ = cv2.findContours(mammogram_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        largest_contour = max(all_contours, key=cv2.contourArea)

        # Create binary mark to remove artifacts
        mammogram_mask = np.zeros(mammogram_threshold.shape, np.uint8)

        cv2.drawContours(mammogram_mask, [largest_contour], -1, 255, cv2.FILLED)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Update boxes
        bounding_boxes = adjust_bounding_boxes(bounding_boxes, x, y)

        # Apply mask to original mammogram
        cropped_mammogram = mammogram[y:y + h, x:x + w]
        cropped_mammogram_mask = mammogram_mask[y:y + h, x:x + w]

        # Set non breast elements to black
        masked_mammogram = cropped_mammogram.copy()
        masked_mammogram[cropped_mammogram_mask == 0] = 0

        # Enhance scans
        clahe_create = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_mammogram = clahe_create.apply(masked_mammogram)

        # Save
        filename = scan.Full_Mammogram_Location.split('\\')[8] + '.png'
        save_mammogram(enhanced_mammogram, target, filename)

        # Validate boxes
        for box in bounding_boxes:
            box[0] = valid_box(box[0])
            box[1] = valid_box(box[1])

        # Set Values
        df.at[scan[0], 'Box_Coords'] = bounding_boxes
        df.at[scan[0], 'Filename'] = filename
        df.at[scan[0], 'Image_Dimensions'] = enhanced_mammogram.shape

    return df


if __name__ == '__main__':
    # Main method run when script is executed

    # Get user arguments
    user_arguments = parse_options()

    print("\n Processing training data")
    training_df = read_dataset_information(os.path.join(user_arguments.data, 'mass_train_data.csv'))
    training_df = preprocessing(training_df.copy(), os.path.join(user_arguments.images, 'train'))
    training_df.to_csv(os.path.join(user_arguments.data, 'mass_train_data_processed.csv'), index=False)

    print("\n Processing testing data")
    testing_df = read_dataset_information(os.path.join(user_arguments.data, 'mass_test_data.csv'))
    testing_df = preprocessing(testing_df.copy(), os.path.join(user_arguments.images, 'test'))
    testing_df.to_csv(os.path.join(user_arguments.data, 'mass_test_data_processed.csv'), index=False)

    print("\n Processing validation data")
    validation_df = read_dataset_information(os.path.join(user_arguments.data, 'mass_val_data.csv'))
    validation_df = preprocessing(validation_df.copy(), os.path.join(user_arguments.images, 'val'))
    validation_df.to_csv(os.path.join(user_arguments.data, 'mass_val_data_processed.csv'), index=False)
