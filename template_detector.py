import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import convolve, gaussian_filter
from math import sqrt
from sklearn.metrics import mean_squared_error
from typing import Union, List
from base_detector import BaseDetector, Centroid


class TemplateDetector(BaseDetector):
    def __init__(self, template_hole_path: str, template_puzle_path: str):
        self.template_hole_path = template_hole_path
        self.template_puzzle_path = template_puzle_path
    
    def detect(self, path, visualize: bool = False) -> Union[List[Centroid], np.ndarray]:

        #output centroid for one image, list of two coordinates one for puzzle [centroid1,centroid2]

        template1 = cv.imread(self.template_puzzle_path,0)
        template2 = cv.imread(self.template_hole_path,0)
        
        img = cv.imread(path,0)
        w1, h1 = template1.shape[::-1]
        w2, h2 = template2.shape[::-1]
        # Apply template Matching
        res1 = cv.matchTemplate(img,template1,cv.TM_CCOEFF)
        res2 = cv.matchTemplate(img,template2,cv.TM_CCOEFF)
        _, _, _, max_loc1 = cv.minMaxLoc(res1)
        _, _, _, max_loc2 = cv.minMaxLoc(res2)
        top_left1 = max_loc1
        top_left2 = max_loc2
        PuzzleCentroid = Centroid(top_left1[0] + w1/2, top_left1[1] + h1/2)
        HoleCentroid = Centroid(top_left2[0] + w2/2,top_left2[1] + h2/2)
        
        if visualize:
            TemplateDetector.visualizeImg(cv.imread(path), PuzzleCentroid, HoleCentroid)
        
        return([PuzzleCentroid,HoleCentroid])
            
    
    @staticmethod
    def visualizeImg(image, PuzzleCentroid: Centroid, HoleCentroid: Centroid):
        image = cv.circle(image,(int(PuzzleCentroid.x),int(PuzzleCentroid.y)), 1, (0, 0, 255), 10)
        image = cv.circle(image,(int(HoleCentroid.x),int(HoleCentroid.y)), 1, (255, 0, 0), 10)
        print('Puzzle Centroid: Blue')
        print('Hole Centroid: Red')
        plt.imshow(image,cmap='gray')
        
# cd = TemplateDetector.detect(path='data/12.png',puzzle_template_path='data/Templates/puzzle2.png',hole_template_path='data/Templates/hole2.png', visualize=True)
