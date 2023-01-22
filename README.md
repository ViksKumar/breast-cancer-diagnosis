# Computer Aided Diagnosis for Breast Cancer

## Abstract 

This project aims to help radiologists in the detection and diagnosis of benign and malignant masses in the breast from mammogram scans to improve the accuracy of mammogram readings. A large open-source dataset of mammogram images is restructured to be more user-friendly before data pre-processing is applied to the images to remove visual artifacts and increase contrast. A custom object detection model is trained on the data and accessed through a graphical user interface developed to load and filter multiple mammograms. The model could be beneficial in a double-reading environment.

## Background and Motivation

Breast cancer is the most common cancer in the UK with over 150 daily diagnoses, making up 15% of all newly diagnosed cancers.<sup>[1](#r1)</sup> Cancer Research UK report that there are 11,500 deaths relating to breast cancer every year and the incidence rate for invasive cancers is set to rise a further 2% leading up to 210 cases per 100,000 women by 2035, having already risen 24% since the 1990s.<sup>[2](#r2)</sup> It is estimated at 23% of breast cancer cases are preventable.<sup>[2](#r2)</sup> The NHS Breast Screening Programme was set up in England in 1988 providing free screening to woman aged 50 up to their 71st birthday. Breast screening captures two x-rays known as mammograms of each breast and results are expected within 2 weeks. Mammograms help to pick up cancers at an early stage, increasing the probability of cure and reducing the number of breast cancer deaths by 1,300 a year in the UK.<sup>[2](#r2)</sup> The workload of radiological services increases by an estimated 10-12% a year without an increase in staff.<sup>[3](#r3)</sup> Workforce fatigue could start to take place which could impact on the quality of diagnosis.<sup>[4](#r4)</sup> The rate of missed diagnosis on initial screening has been as high as 46.6%.<sup>[5](#r5)</sup> Double reading is an approach aimed at reducing the number of missed diagnoses by having two radiologists analyse a scan. However, given the problem of fatigue and increasing workload spending extra resources on double reading might not be an optimal approach. Computer Aided Diagnosis (CAD) could be the solution to this problem and could become an increasingly important tool in early diagnosis. The development of CAD has also seen with it a rise in associated legal issues. Decisions made by CAD are subject to GDPR restrictions and article 22 outlines these restrictions for automated decision making.<sup>[6](#r6)</sup> In the context of CAD, decisions must involve human contribution unless specific consent is given, or the process is authorised by law. This reinforces the idea of CAD in mammography for second reading, rather than as a primary diagnosis tool.

## Dataset Preparation

<p align="center">
    <img 
         src="https://github.com/ViksKumar/images/blob/fa9a55ab24a461820da94575d47d386c087b03bc/data%20engineering%20stages.png" 
         height="700px"
         alt="A data flow diagram for CBIS-DDSM"
    >
</p>
<p align="center">
<em>Figure 1. A data flow diagram for CBIS-DDSM</em>
</p>


The Curated Breast Imaging Subset of DDSM dataset<sup>[7](#r7)</sup> was chosen to train the tumor detection model as it contains a large sample size of breast mammograms with benign and malignant cases. The original structure of the dataset contains many nested folders. The root of the directory contains separate folders for each full mammogram and each individual tumour. 
Inside these folders there are further nested folders before the file root is eventually reached where, in the case of a full mammogram, there will be a singular DICOM file, but in the case of a tumour file there will be two DICOM files, one for the binary ground truth and another for the cropped ROI. The DICOM filenames are numbered but this numbering is inconsistent. A cropped ROI might be named the same as a ground truth file in another folder. This format is unintuitive and difficult to work with since data is not grouped logically by individual mammogram and associated regions of interest. A batch sequential architecture will first separate the CBIS-DDSM data into a usable format. A metadata file is generated when extracting the dataset from the source and contains the filename and local file paths as information of interest. The information file for training and testing provided with the CBIS-DDSM contains detailed information for each mammogram in the appropriate dataset but not where it is located. The test set provided will be further split into a testing and validation set with a 50% split. The result of the data engineering stage is a novel solution for processing of the complicated CBIS-DDSM dataset to produce a single csv file for training, testing and validation information containing all of the needed information for developing a model and sorting the file structure.


## References

1. <a name="r1">Cancer Research UK. What is Breast Cancer. 2021 [cited 2022 April 22]. Available from: https://www.cancerresearchuk.org/about-cancer/breast-cancer/about</a>
2. <a name="r2">Cancer Research UK. Breast cancer statistics. N.d. [cited 2022 April 22]. Available from: https://www.cancerresearchuk.org/health-professional/cancer-statistics/statistics-by-cancer-type/breast-cancer#heading-Zero</a>
3. <a name="r3">Royal College of Radiologists. Clinical radiology UK workforce census 2015 report. . 2016 [cited 2022 April 22].</a>
4. <a name="r4">Taylor-Phillips S, Stinton C. Fatigue in radiology: a fertile area for future research. 2019 [cited 2022 April 22]. Available from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6636274/#b6</a>
5. <a name="r5">Waheed KB, Hassan M, Hassan DA, Shamrani A, Bassam MA, Elbyali AA, Shams TM, Demiati ZA, Arulanatham ZJ. Breast cancers missed during screening in a tertiary-care hospital mammography facility. 2019. [cited 2022 April 22] Available from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6838646/</a>
6. <a name="r6">European Parliament and Council of the European Union. Art. 22 GDPR. 2018 [cited 2022 April 22]. Available from: https://gdpr-info.eu/art-22-gdpr/</a>
7. <a name="r7">Lee RS, Gimenez F, Hoogi A, Rubin D. Curated Breast Imaging Subset of DDSM . 2016 [cited 2022 April 22]. Available from: https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM</a>

