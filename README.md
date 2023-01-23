# Computer Aided Diagnosis for Breast Cancer

## Abstract 

This project aims to help radiologists in the detection and diagnosis of benign and malignant masses in the breast from mammogram scans to improve the accuracy of mammogram readings. A large open-source dataset of mammogram images is restructured to be more user-friendly before data pre-processing is applied to the images to remove visual artifacts and increase contrast. A custom object detection model is trained on the data and accessed through a graphical user interface developed to load and filter multiple mammograms. The model could be beneficial in a double-reading environment.

## Background and Motivation

Breast cancer is the most common cancer in the UK with over 150 daily diagnoses, making up 15% of all newly diagnosed cancers.<sup>[1](#r1)</sup> Cancer Research UK report that there are 11,500 deaths relating to breast cancer every year and the incidence rate for invasive cancers is set to rise a further 2% leading up to 210 cases per 100,000 women by 2035, having already risen 24% since the 1990s.<sup>[2](#r2)</sup> It is estimated at 23% of breast cancer cases are preventable.<sup>[2](#r2)</sup> The NHS Breast Screening Programme was set up in England in 1988 providing free screening to woman aged 50 up to their 71st birthday. Breast screening captures two x-rays known as mammograms of each breast and results are expected within 2 weeks. Mammograms help to pick up cancers at an early stage, increasing the probability of cure and reducing the number of breast cancer deaths by 1,300 a year in the UK.<sup>[2](#r2)</sup> The workload of radiological services increases by an estimated 10-12% a year without an increase in staff.<sup>[3](#r3)</sup> Workforce fatigue could start to take place which could impact on the quality of diagnosis.<sup>[4](#r4)</sup> The rate of missed diagnosis on initial screening has been as high as 46.6%.<sup>[5](#r5)</sup> Double reading is an approach aimed at reducing the number of missed diagnoses by having two radiologists analyse a scan. However, given the problem of fatigue and increasing workload spending extra resources on double reading might not be an optimal approach. Computer Aided Diagnosis (CAD) could be the solution to this problem and could become an increasingly important tool in early diagnosis. The development of CAD has also seen with it a rise in associated legal issues. Decisions made by CAD are subject to GDPR restrictions and article 22 outlines these restrictions for automated decision making.<sup>[6](#r6)</sup> In the context of CAD, decisions must involve human contribution unless specific consent is given, or the process is authorised by law. This reinforces the idea of CAD in mammography for second reading, rather than as a primary diagnosis tool.

## Project Files:
- [**ddsm_handling.py**](https://github.com/ViksKumar/breast-cancer-diagnosis/blob/19f23ebe1aa080913cb47b4b172d55611df0b832/ddsm_handling.py): Restructures the CBIS-DDSM dataset making it easier to use
- [**data_preprocessing.py**](https://github.com/ViksKumar/breast-cancer-diagnosis/blob/19f23ebe1aa080913cb47b4b172d55611df0b832/data_preprocessing.py): Preprocesses mammogram images for model training
- [**yolo_labels_creation.py**](https://github.com/ViksKumar/breast-cancer-diagnosis/blob/19f23ebe1aa080913cb47b4b172d55611df0b832/yolo_labels_creation.py): Labels data in a format compatible with Yolov5 for supervised learning 
- [**solution_gui.py**](https://github.com/ViksKumar/breast-cancer-diagnosis/blob/19f23ebe1aa080913cb47b4b172d55611df0b832/solution_gui.py): A GUI to use the trained model
- [**requirements.txt**](https://github.com/ViksKumar/breast-cancer-diagnosis/blob/19f23ebe1aa080913cb47b4b172d55611df0b832/requirements.txt): A list of the project's dependencies
- [**best.pt**](https://github.com/ViksKumar/breast-cancer-diagnosis/blob/19f23ebe1aa080913cb47b4b172d55611df0b832/best.pt): Trained model file useable with PyTorch



## Dataset Preparation

<a name="f1">
<p align="center">
    <img 
         src="https://github.com/ViksKumar/images/blob/fa9a55ab24a461820da94575d47d386c087b03bc/data%20engineering%20stages.png" 
         height="700px"
         alt="A data flow diagram for CBIS-DDSM"
    >
</p>
<p align="center">
<em>Figure 1. Data flow diagram showing processing applied to CBIS-DDSM</em>
</p>
</a>

The Curated Breast Imaging Subset of DDSM dataset<sup>[7](#r7)</sup> was chosen to train the tumor detection model as it contains a large number of breast mammograms with benign and malignant cases. However, I found using this dataset raw unintuitive for this project for a number of reasons:

- File naming conventions for the ground truth masks are inconsistent between cases 
- The structure of the dataset contains an unnecessary number of nested folders
- Mammograms have the DICOM filetype, which is difficult to work with
- The dataset contains a number of unneeded fields
- Local filepaths are missing

The [ddsm_handling.py](https://github.com/ViksKumar/breast-cancer-diagnosis/blob/19f23ebe1aa080913cb47b4b172d55611df0b832/ddsm_handling.py) script was written to overcome these issues. [Figure 1.](#f1) shows the steps taken to process the dataset. 



## Image pre-processing

<a name="f2">
<p align="center">
    <img 
         src="https://github.com/ViksKumar/images/blob/fa9a55ab24a461820da94575d47d386c087b03bc/preprocessing%20stages.png"
         height="400px"
         alt="A data flow diagram for CBIS-DDSM"
    >
</p>
<p align="center">
<em>Figure 2. Flow diagram for mammogram pre-processing</em>
</p>
</a>

The aim of the pre-processing is to optimise the images to improve the training data and produce higher quality results. The mammograms in the CBIS-DDSM contain visual-noise such as annotations and tape marks from the original sampling. The mammograms also contain large areas of background which will take up additional processing time. The [data_preprocessing.py](https://github.com/ViksKumar/breast-cancer-diagnosis/blob/19f23ebe1aa080913cb47b4b172d55611df0b832/data_preprocessing.py) script improves the quality of the mammograms by removing unwanted noise and cropping the mammogram around the breast to remove excess background area. Additionally, the mammograms will be enhanced to improve contrast between intense tumour areas and more subtle background tissue. The bounding box co-ordinates for each of the tumours in a mammogram are calculated from the provided ground truth masks. Subsequently, the [yolo_labels_creation.py](https://github.com/ViksKumar/breast-cancer-diagnosis/blob/19f23ebe1aa080913cb47b4b172d55611df0b832/yolo_labels_creation.py) script is used to convert the labels for the bounding boxes into the format required by yolov5 and produce label and image files for each of the mass mammograms in the CBIS-DDSM dataset. 

<a name="f3">
<p align="center">
    <img 
         src="https://github.com/ViksKumar/images/blob/fa9a55ab24a461820da94575d47d386c087b03bc/preprocessing_stages.png"
         height="300px"
         alt="Figure 3. Pre-processing mammogram stages"
    >
</p>
<a name="f3">

<p align="center">
<em>Figure 3. Pre-processing mammogram stages</em>
</p>

## Model training

The tumour detection model was trained using pretrained weights from the yolov5s model and training scripts available from the [yolov5 repository](https://github.com/ultralytics/yolov5). Image size, batch size and number of epochs are all kept constant for comparisons. Training was performed with an image size of 640 and batch-size of 16 for 20 epochs. The model weights, batch-size and image size were all limited by hardware. Increasing these values further/using a larger model would cause the training process to run out of available memory.  Yolov5 has tuneable model hyperparameters, many of which relate to data augmentation. These were tuned manually with a trial-and-error approach as an exhaustive search method, such as Yolov5’s evolve function, was unable to be computed in an appropriate time. Data augmentation parameters were tuned to flip the image along the x and y axis, in addition to rotating the image. Further distortions were not applied as these would not produce realistic mammogram images.  The model was trained on 1318 samples, where 637 were malignant and 681 were benign. The validation set consisted of 184 mammograms, 117 benign cases and 67 malignant cases. The test set contains 192 unseen mammograms with 114 benign cases and 78 malignant cases. 


## Graphical User Interface
    
The GUI enables users to load and detect tumours in mammograms and save report files for batch processing. 


    
[<img alt="Click to view GUI demonstration video" src="https://github.com/ViksKumar/images/blob/fa9a55ab24a461820da94575d47d386c087b03bc/gui.png" />](https://www.youtube.com/watch?v=Ecs8IeH-dPQ)
<br><em>Click the image to view the GUI demonstation video</em>
 
    
## Difficult cases
    
The model evaluations showed a specific problem with benign cases. To further understand why, a benign tumour that was missed during testing was compared with a correctly diagnosed malignant and benign tumour in addition to a random sample of normal tissue. (Figure 4) To compare these segments the hamming distance was calculated between the misdiagnosed benign tumour and the three correctly classified segments. The hamming distance represents the number of positions in which the symbols of two equal length hashed strings formed from the images are different. The lower the hamming distance the more similar the strings are. The hamming distances show that the missed benign tumour is most similar to normal tissue, rather than other benign or malignant tumours. Hence the diagnosis is completely missed, and sensitivity is reduced. The difference in hamming distance to the detected malignant and benign tumours is very small. This could lead to misclassification even if the tumour is correctly localised. Further work could be carried out to better pre-process benign cases to make them more distinguishable from other classifications.

<a name="f4">
<p align="center">
    <img 
         src="https://github.com/ViksKumar/images/blob/fa9a55ab24a461820da94575d47d386c087b03bc/segment%20comparisons.png"
         height="400px"
         alt="Segment Comparisons"
    >
</p>
</a>
<p align="center">
<em>Figure 4. Segment Comparisons for Tumours</em>
</p>
    
## References

1. <a name="r1">Cancer Research UK. What is Breast Cancer. 2021 [cited 2022 April 22]. Available from: https://www.cancerresearchuk.org/about-cancer/breast-cancer/about</a>
2. <a name="r2">Cancer Research UK. Breast cancer statistics. N.d. [cited 2022 April 22]. Available from: https://www.cancerresearchuk.org/health-professional/cancer-statistics/statistics-by-cancer-type/breast-cancer#heading-Zero</a>
3. <a name="r3">Royal College of Radiologists. Clinical radiology UK workforce census 2015 report. . 2016 [cited 2022 April 22].</a>
4. <a name="r4">Taylor-Phillips S, Stinton C. Fatigue in radiology: a fertile area for future research. 2019 [cited 2022 April 22]. Available from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6636274/#b6</a>
5. <a name="r5">Waheed KB, Hassan M, Hassan DA, Shamrani A, Bassam MA, Elbyali AA, Shams TM, Demiati ZA, Arulanatham ZJ. Breast cancers missed during screening in a tertiary-care hospital mammography facility. 2019. [cited 2022 April 22] Available from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6838646/</a>
6. <a name="r6">European Parliament and Council of the European Union. Art. 22 GDPR. 2018 [cited 2022 April 22]. Available from: https://gdpr-info.eu/art-22-gdpr/</a>
7. <a name="r7">Lee RS, Gimenez F, Hoogi A, Rubin D. Curated Breast Imaging Subset of DDSM . 2016 [cited 2022 April 22]. Available from: https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM</a>

