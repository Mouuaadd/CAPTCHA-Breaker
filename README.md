# CAPTCHA solver with Computer Vision

Welcome to the CAPTCHA solver with Computer Vision repository! This repository contains the files for the project assignment in the Visual Computing course at CentraleSupelec.


<img src="https://user-images.githubusercontent.com/81930893/231488706-0d6b5d4e-8fb2-4306-bce6-fdcda454fd69.JPG" width="500">


## Annotations

Format object keys:

- image: path
- annotation: List:
  - label: label_name
  - coordinates: List:
    - x: float
    - y: float
    - height: float
    - width: float

## Methods

The goal of the homework is to implement various detector algorithms using the base detector provided in `base_detector.py`.

CAPTCHA solving is done with one of these methods:

1. Edge Detection
2. Template Matching
3. Color Segmentation
4. Bag of SIFTs descriptors
5. Hough Transform
6. Harris corner detector

## Results

<img src="https://user-images.githubusercontent.com/81930893/231488637-590e0cc0-488d-4ff3-b74e-3a00afeb2cdc.JPG" width="500">
<img src="https://user-images.githubusercontent.com/81930893/231489299-b9297d0d-1be7-4b3c-88cb-2a8de4df4bd9.JPG" width="500">



To run this project, open the `run.ipynb`, uncomment the initialization of the detector you want to use, then run the cell. 
