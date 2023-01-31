# CAPTCHA solver with Computer Vision

Welcome to the CAPTCHA solver with Computer Vision repository! This repository contains the files for the project assignment in the Visual Computing course at CentraleSupelec.

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

To run this project, open the `run.ipynb`, uncomment the initialization of the detector you want to use, then run the cell. 
