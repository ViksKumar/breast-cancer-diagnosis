# A Python Script to process CBIS-DDSM information by Vikram Kumar
"""
Create easy to use training, testing and validation data from the extracted and installed CBIS-DDSM dataset

Original Data by Rebecca Sawyer Lee, Francisco Gimenez, Assaf Hoogi , Daniel Rubin  (2016).
Curated Breast Imaging Subset of DDSM [Dataset]. The Cancer Imaging Archive.
DOI:  https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY

Usage:
    $ python path/to/DDSM_Processing.py --test "mass_case_description_test_set.csv"
                                        --train "mass_case_description_train_set.csv"
                                        --meta "metadata.csv"
                                        --images "CBIS-DDSM\manifest-ZkhPvrLo5216730872708713142"
                                        --output "CBIS-DDSM\processed"
"""

import sys
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
import argparse


def parse_options():
    """
    Parse the arguments from the user input in the command line which contains filepaths
    :return: arguments from user input
    """
    parser = argparse.ArgumentParser(description='Reformat CBIS-DDSM Dataset')
    parser.add_argument('--test', required=True, help='filepath for CBIS-DDSM test dataset')
    parser.add_argument('--train', required=True, help='filepath for CBIS-DDSM train dataset')
    parser.add_argument('--meta', required=True, help='filepath for CBIS-DDSM download metadata')
    parser.add_argument('--images', required=True, help='directory of extracted dataset')
    parser.add_argument('--output', required=True, help='folder where datasets will be saved')
    args = parser.parse_args()
    return args


def join_meta_paths(original_path, image_path):
    """
    Join the file locations from the metadata to the local installation directory
    :param original_path: (string) filepath from metadata file
    :param image_path: (string) filepath from user where dataset is saved
    :return: (string) combined filepath
    """
    # Need to remove '.' to form valid filepath, leading '\' stripped to ensure join includes both filepaths
    original_path = original_path.split('.', 1)[1].lstrip('\\')
    return os.path.join(image_path, original_path)


def restructure_metadata(metadata_df, image_location):
    """
    Remove unneeded columns from metadata dataframe and edit filepaths to include local directory
    :param metadata_df: (pandas dataframe) metadata information
    :param image_location: (string) filepath where CBIS-DDSM was extracted
    :return: (none) changes in function made directly to original dataframe
    """
    metadata_df['File Location'] = metadata_df.apply(lambda x: join_meta_paths(x['File Location'],
                                                                               image_location), axis=1)
    metadata_df = metadata_df[['Subject ID', 'File Location', 'Series Description']]


def join_full_paths(original_path):
    """
    Create filepath with specific filename for full mammogram images
    :param original_path: (string) full mammogram folder
    :return: (string) full mammogram filepath including filename
    """
    # Filename of full mammogram images is standard
    return os.path.join(original_path, '1-1.dcm')


def restructure_full_mammograms(metadata_df):
    """
    Create dataframe from a copy of the metadata to include full mammogram filepaths
    :param metadata_df: (pandas dataframe) metadata information
    :return: (pandas dataframe) dataframe with precise filepaths for full mammogram images
    """
    df = metadata_df.copy()

    # Filter dataframe to only contain data for full mammogram images
    df = df[df['Series Description'] == 'full mammogram images']
    df['File Location'] = df['File Location'].apply(join_full_paths)
    return df


def restructure_roi(metadata_df, mammogram_df):
    """
    Create a dataframe containing ROI mask information from a copy of the metadata dataframe
    :param metadata_df: (pandas dataframe) metadata information
    :param mammogram_df: (pandas dataframe) mammogram information
    :return: (pandas dataframe) ROI information
    """

    # Work on copies
    metadata_df_copy = metadata_df.copy()
    roi_df = mammogram_df.copy()

    # Filter dataframe to only contain data for ROI mask images
    roi_images = metadata_df_copy[metadata_df_copy['Series Description'] == 'ROI mask images']

    # Remove unneeded columns
    roi_df = roi_df[['pathology', 'image file path', 'cropped image file path']]

    # Split filepaths to get the root folder names
    roi_df['image file path'] = roi_df['image file path'].str.split('/').str[0]
    roi_df['cropped image file path'] = roi_df['cropped image file path'].str.split('/').str[0]

    # Merge the ROI mask and metadata dataframes
    merged_data = pd.merge(roi_df, roi_images, how='inner', left_on='cropped image file path',
                           right_on='Subject ID')

    # Keep columns containing classification, filepath of full mammogram images and filepath of ROI mask.
    merged_data = merged_data[['pathology', 'image file path', 'File Location']]
    merged_data.rename(columns={'pathology': 'Class', 'File Location': 'Mask_Location'}, inplace=True)

    # Classify all benign without callback cases as benign to create a binary classification
    merged_data['Class'].replace(to_replace='BENIGN_WITHOUT_CALLBACK', value='BENIGN', inplace=True)

    return merged_data


def encode(train_df, test_df):
    """
    Label encode the classification for use with object detection models
    :param train_df: (pandas dataframe) training data information
    :param test_df: (pandas dataframe) testing data information
    :return: (none) changes made in function directly to original dataframes
    """

    # Fit label encoder on the training data
    le = preprocessing.LabelEncoder()
    le.fit(train_df['Class'])

    # Transform the training and test data using the same label encoder to ensure encoding is consistent
    train_df['Class'] = le.transform(train_df['Class'])
    test_df['Class'] = le.transform(test_df['Class'])


def validate_path(input_path):
    """
    Validate user inputs by reading the specified files
    :param input_path: (string) filepath parsed from user input
    :return: (none) scripts ends if error is raised
    """
    try:
        pd.read_csv(input_path)
    except:
        print(f"Could not read {input_path}, exiting...")
        sys.exit()


def restructure_group(roi_df, full_df):
    """
    Produce a final dataframe by combining ROI mask and full mammogram information
    :param roi_df: (pandas dataframe) ROI mask information
    :param full_df: (pandas dataframe) Full mammogram information
    :return: (pandas dataframe) Merged dataframe grouped by each full mammogram
    """
    final_df = pd.merge(roi_df, full_df, how='left', left_on='image file path', right_on='Subject ID')
    final_df = final_df[['Class', 'Mask_Location', 'File Location']]
    final_df.rename(columns={'File Location': 'Full_Mammogram_Location'}, inplace=True)

    # Each full mammogram image can have multiple corresponding ROI masks/classifications, group by each full mammogram
    final_df = final_df.groupby(['Full_Mammogram_Location']).agg(tuple).applymap(list).reset_index()
    return final_df


def save_datasets(train_df, test_df, target):
    """
    Create a validation set and save the processed CBIS-DDSM information to .csv files
    :param train_df: (pandas dataframe) training data information
    :param test_df: (pandas dataframe) testing data information
    :param target: (string) filepath where .csv files will be saved
    :return: (none) function saves files
    """

    # Split the testing set into half to create a validation set
    val, test = train_test_split(test_df, test_size=0.5, random_state=1)

    # Save processed information
    val.to_csv(os.path.join(target, 'mass_val_data.csv'), index=False)
    test.to_csv(os.path.join(target, 'mass_test_data.csv'), index=False)
    train_df.to_csv(os.path.join(target, 'mass_train_data.csv'), index=False)


if __name__ == '__main__':
    # Main method run when script is executed

    # Get user arguments
    user_arguments = parse_options()

    # Check user specified files exist
    filepaths = [user_arguments.train, user_arguments.test, user_arguments.meta]
    for path in filepaths:
        validate_path(path)

    # Read user specified csv files into pandas dataframes
    metadata_df = pd.read_csv(user_arguments.meta, index_col=None)
    train_df = pd.read_csv(user_arguments.train, index_col=None)
    test_df = pd.read_csv(user_arguments.test, index_col=None)

    # Restructure and edit metadata information
    restructure_metadata(metadata_df, user_arguments.images)

    # Get full mammogram information from metadata
    full_mammogram_df = restructure_full_mammograms(metadata_df)

    # Get ROI mask information for training and testing datasets
    roi_mammogram_df_train = restructure_roi(metadata_df, train_df)
    roi_mammogram_df_test = restructure_roi(metadata_df, test_df)

    # Label encode data
    encode(roi_mammogram_df_train,  roi_mammogram_df_test)

    # Create a final grouped dataframe
    final_df_train = restructure_group(roi_mammogram_df_train.copy(), full_mammogram_df.copy())
    final_df_test = restructure_group(roi_mammogram_df_test.copy(), full_mammogram_df.copy())

    # Save processed information
    save_datasets(final_df_train, final_df_test, user_arguments.output)
    print(f'Processed data and saved information in: {user_arguments.output}')
