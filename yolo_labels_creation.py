# A script to create the label files required by yolov5
import pandas as pd
from ast import literal_eval
import os

target = r'C:\Users\Vikram\Documents\FYP\CBIS-DDSM\labels\val'
information = pd.read_csv(r'C:\Users\Vikram\Documents\FYP\CBIS-DDSM\mass_val_data_processed.csv', index_col=None)
information.Class = information.Class.apply(literal_eval)
information.Mask_Location = information.Mask_Location.apply(literal_eval)
information.Box_Coords = information.Box_Coords.apply(literal_eval)
information.Image_Dimensions = information.Image_Dimensions.apply(literal_eval)


for scan in information.itertuples():
    os.makedirs(target, exist_ok=True)
    yolo_filename = scan.Filename.split('.')[0] + '.txt'
    yolo_path = os.path.join(target, yolo_filename)
    with open(yolo_path, 'w') as f:
        for (label, boxes) in zip(scan.Class, scan.Box_Coords):
            # Bbox co-ordinates as per the format required by YOLO v5
            xmin = boxes[0]
            xmax = boxes[0] + boxes[2]
            ymin = boxes[1]
            ymax = boxes[1] + boxes[3]

            # Box Coords
            Box_Center_X = ((xmin + xmax)/2)
            Box_Center_Y = ((ymin + ymax)/2)
            Box_Width = boxes[2]
            Box_Height = boxes[3]

            # Image Size
            Image_Height, Image_Width = scan.Image_Dimensions

            # Normalise Coords
            Box_Center_X = Box_Center_X / Image_Width
            Box_Center_Y = Box_Center_Y / Image_Height
            Box_Width = Box_Width / Image_Width
            Box_Height = Box_Height / Image_Height
            f.write(f'{label} {Box_Center_X:.3f} {Box_Center_Y:.3f} {Box_Width:.3f} {Box_Height:.3f}\n')


