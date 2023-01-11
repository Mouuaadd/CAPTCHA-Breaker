import numpy as np
import cv2
from typing import List, Tuple, Union

from base_detector import BaseDetector, Centroid


class EdgeDetector(BaseDetector):
    """
    A class for detecting edges in an image using the Sobel operator and
    finding intersections of those edges.

    Attributes:
        None
    """

    def __init__(self):
        """
        Initializes an EdgeDetector object.
        """
        pass

    def detect(self, path, visualize: bool = False) -> Union[List[Centroid], np.ndarray]:
        """
        Detects edges and intersections in the image at the specified path.

        Args:
            path (str): The filepath of the image to be processed.

        Returns:
            list: A list of tuples representing the (x, y) coordinates of
                the detected intersections.
        """
        visualization, centroids = EdgeDetector.full_edge_detection_pipe(
            path)
        if visualize:
            return visualization
        return centroids

    @staticmethod
    def get_intersections(edges, min_length):
        """
        Detects intersections in an image represented by a matrix of edge pixels.

        Parameters:
            edges (ndarray): A matrix of edge pixels in the image.
            min_length (int): The minimum length of a line in the image to be considered an intersection.

        Returns:
            list: A list of tuples representing the x, y coordinates of the detected intersections.
        """
        y, x = np.where(edges != 0)
        uniques_x = np.unique(x)
        uniques_y = np.unique(y)
        line_x = []
        line_y = []
        for unique_x in uniques_x:
            points = np.where(edges[:, unique_x] != 0)
            if points[0].shape[0] >= min_length:
                line_y.append(unique_x)
        for unique_y in uniques_y:
            points = np.where(edges[unique_y, :] != 0)
            if points[0].shape[0] >= min_length:
                line_x.append(unique_y)
        intersections = []
        for y_coord in line_x:
            x_coords = np.where(edges[y_coord, :])[0]
            for x_point in line_y:
                for x_coord in x_coords:
                    if x_coord == x_point:
                        intersections.append((x_coord, y_coord))
        return intersections

    @staticmethod
    def apply_sobel(image, threshold=140):
        """
        Applies the Sobel operator to an image to detect horizontal and vertical edges.

        Parameters:
            image (ndarray): The image to apply the Sobel operator to.
            threshold (int): The threshold for determining which pixels are considered edges.

        Returns:
            ndarray: A matrix of edge pixels in the image.
        """
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply the Sobel operator to detect horizontal edges
        horizontal_edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

        # Apply the Sobel operator to detect vertical edges
        vertical_edges = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Combine the detected horizontal and vertical edges
        edges = cv2.addWeighted(horizontal_edges, 0.5, vertical_edges, 0.5, 0)

        edges[edges <= threshold] = 0
        edges[edges >= threshold] = 255
        return edges

    @staticmethod
    def apply_threshold(matrix, thresh):
        matrix[matrix >= thresh] = 255
        matrix[matrix <= thresh] = 0
        return matrix

    @staticmethod
    def get_centroids(intersections: List[Tuple[int, int]]) -> List[Centroid]:
        top_left, bot_right = intersections
        length = bot_right[1] - top_left[1]
        new_top_left = [top_left[0] + length // 2, top_left[1] + length // 2]
        new_bot_right = [bot_right[0] - length //
                         2, bot_right[1] - length // 2]
        return [
            Centroid(*new_top_left),
            Centroid(*new_bot_right)
        ]

    @staticmethod
    def filter_intersections(intersections, margin=100):
        top_left_coord = [float("inf"), float("inf")]
        bot_right_coord = [-float("inf"), -float("inf")]
        for point in intersections:
            top_left_coord[0] = min(top_left_coord[0], point[0])
            top_left_coord[1] = min(top_left_coord[1], point[1])
        for point in intersections:
            if point[0] > bot_right_coord[0]:
                bot_right_coord[0] = point[0]
            if point[1] > top_left_coord[0] and point[1] < top_left_coord[1] + margin:
                bot_right_coord[1] = point[1]
        intersections = [top_left_coord, bot_right_coord]
        return intersections

    @staticmethod
    def full_edge_detection_pipe(path):
        """
        Applies the full edge detection pipeline to an image at the specified path.

        Parameters:
            path (str): The path to the image.

        Returns:
            tuple: A tuple containing the image with intersections marked and a list of tuples representing the x, y coordinates of the detected intersections.
        """
        image = cv2.imread(path)
        sobeled_img = EdgeDetector.apply_sobel(image)
        intersections = EdgeDetector.get_intersections(sobeled_img, 40)
        intersections = EdgeDetector.filter_intersections(intersections)
        centroids = EdgeDetector.get_centroids(intersections)
        for coord in centroids:
            cv2.circle(image, (coord.x, coord.y), 5, (255), 2)
        return image, intersections
