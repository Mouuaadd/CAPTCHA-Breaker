import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, gaussian_filter
from math import sqrt
from sklearn.metrics import mean_squared_error
from typing import Union, List
from base_detector import BaseDetector, Centroid


class ColorDetector(BaseDetector):
    def __init__(self):
        pass

    def detect(self, path: str, visualize: bool = False) -> Union[List[Centroid], np.ndarray]:

        # output centroid for one image, list of two coordinates one for puzzle [centroid1,centroid2]

        Puzzle_lower_rgb = (140, 184, 60)
        Puzzle_upper_rgb = (190, 255, 165)

        Hole_lower_rgb = (0, 0, 28)
        Hole_upper_rgb = (80, 62, 57)

        se = np.ones((2, 2))

        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.GaussianBlur(img, ksize=(3, 3), sigmaX=5, sigmaY=5)
        mask1 = cv.inRange(img, Puzzle_lower_rgb, Puzzle_upper_rgb)
        mask2 = cv.inRange(img, Hole_lower_rgb, Hole_upper_rgb)
        result1 = cv.bitwise_and(img, img, mask=mask1)
        result2 = cv.bitwise_and(img, img, mask=mask2)
        for _ in range(2):
            result1 = cv.dilate(result1, se)
            result2 = cv.dilate(result2, se)
        x_mean, y_mean = cv.findNonZero(cv.cvtColor(
            result1, cv.COLOR_RGB2GRAY)).squeeze().mean(0)
        if cv.findNonZero(cv.cvtColor(result2, cv.COLOR_RGB2GRAY)) is None:
            x_mean_hole, y_mean_hole = (0, 0)
        else:
            x_mean_hole, y_mean_hole = cv.findNonZero(
                cv.cvtColor(result2, cv.COLOR_RGB2GRAY)).squeeze().mean(0)

        PuzzleCentroid = Centroid(x_mean, y_mean)
        HoleCentroid = Centroid(x_mean_hole, y_mean_hole)

        if visualize:
            ColorDetector.visualizeImg(img, PuzzleCentroid, HoleCentroid)

        return [PuzzleCentroid, HoleCentroid]

    def visualizeImg(image, PuzzleCentroid: Centroid, HoleCentroid: Centroid):
        image = cv.circle(image, (int(PuzzleCentroid.x), int(
            PuzzleCentroid.y)), 1, (0, 0, 255), 10)
        image = cv.circle(image, (int(HoleCentroid.x), int(
            HoleCentroid.y)), 1, (255, 0, 0), 10)
        plt.imshow(image)

# cd = ColorDetector.detect(path='data/0.png', visualize=True)
