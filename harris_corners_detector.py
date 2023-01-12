import numpy as np
import cv2
from typing import List, Tuple, Union
from skimage.feature import corner_peaks, corner_harris

from base_detector import BaseDetector, Centroid

from utils.rectanglesFinder import (
    find_squares_and_rectangles,
    Point,
)
from utils.clusterPoints import cluster_points


class HarrisCornersDetector(BaseDetector):
    """
    A class for detecting corners in an image.

    Attributes:
        None
    """

    def __init__(self):
        """
        Initializes a HarrisCornersDetector object.
        """
        pass

    def detect(
        self, path, visualize: bool = False
    ) -> Union[List[Centroid], np.ndarray]:
        """
        Detects edges and intersections in the image at the specified path.

        Parameters:
            path (str): The filepath of the image to be processed.

        Returns:
            list: A list of tuples representing the (x, y) coordinates of
                the detected intersections.
        """
        visualization, centroids = HarrisCornersDetector.full_corners_detection_pipe(
            path
        )
        if visualize:
            return visualization
        return centroids

    @staticmethod
    def apply_preprocessing(image, blur=False, threshold=20):
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

        # Apply the Laplacian operator
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Gaussian blur the image if requested
        if blur:
            laplacian = cv2.GaussianBlur(laplacian, (7, 7), 0)

        # Apply binary thresholding
        binary = cv2.threshold(laplacian, threshold, 255, cv2.THRESH_BINARY)[1]

        return binary

    @staticmethod
    def detect_corners(image, min_distance=10):
        """
        Detects corners in an image using Harriq Corners Detector.

        Parameters:
            image (ndarray): The image to apply the Harris corner detector to.
            min_distance (int): The minimum distance between two corners.

        Returns:
            ndarray: A matrix of corners coordinates in the image.
        """
        # Define the minimum distance between two corners
        min_distance = 10

        # Compute the corner response using the Harris corner detector
        corner_response = corner_harris(image=image, method="k", k=0.04, sigma=3)

        # Find the peaks in the corner response using the corner_peaks function
        corners = corner_peaks(
            corner_response,
            min_distance=min_distance,
            threshold_rel=0.2,
            exclude_border=True,
        )

        if len(corners) < 8:
            raise RuntimeError("Not enough corners")

        # Print the number of corners detected
        print(f"Number of corners detected: {corners.shape[0]}")

        return corners

    @staticmethod
    def rectangles_filter(
        corners, distance_min=40, distance_max=130, length_delta=40, angle_delta=2
    ):
        """
        Filters out points that form rectangles from a list of corners coordinates.

        Parameters:
            corners (list): A list of corners.
            distance_min (int): The minimum distance between two corners.
            distance_max (int): The maximum distance between two corners.
            length_delta (int): The maximum difference between the length of two sides of a rectangle.
            angle_delta (int): The maximum difference between the angle of two sides of a rectangle.s

        Returns:
            rectangles: A list of rectangles.
        """
        # Create a list of Point objects from a list of x,y coordinates
        points = [Point(x, y) for y, x in corners]

        # Find squares and rectangles in the set of points
        rectangles = find_squares_and_rectangles(
            points, distance_min, distance_max, length_delta, angle_delta
        )

        # Unflatten rectangles list
        rectangles = np.ravel(rectangles).tolist()

        print(f"Nb of rectangles found:{len(rectangles)}")

        return rectangles

    @staticmethod
    def get_centroids(points: List[Tuple[int, int]], n_clusters=2) -> List[Centroid]:
        """
        Computes the centroids of a list of points.

        Parameters:
            points (list): A list of points.
            n_clusters (int): The number of clusters to use.

        Returns:
            List[Centroid]: A list of centroids.

        """

        # Convert to numpy array
        Px = np.array([p.x for p in points])
        Py = np.array([p.y for p in points])
        points = np.float32(np.column_stack((Px, Py)))

        centroids = cluster_points(points, n_clusters)

        return centroids

    @staticmethod
    def full_corners_detection_pipe(path):
        """
        Performs the full corners detection pipeline.

        Parameters:
            path (str): The filepath of the image to be processed.

        Returns:
            ndarray: A matrix of edge pixels in the image.
        """
        # Read the image
        image = cv2.imread(path)

        # Apply preprocessing
        binary = HarrisCornersDetector.apply_preprocessing(image, blur=True)

        # Detect corners
        corners = HarrisCornersDetector.detect_corners(binary)

        # Filter out rectangles
        rectangles = HarrisCornersDetector.rectangles_filter(corners)

        # Compute centroids
        centroids = HarrisCornersDetector.get_centroids(rectangles)

        centroid_puzzle_piece = Centroid(*centroids[0])
        centroid_hole = Centroid(*centroids[1])

        # Visualize the results
        cv2.circle(
            image, (centroid_puzzle_piece.x, centroid_puzzle_piece.y), 5, (255), 2
        )
        cv2.circle(image, (centroid_hole.x, centroid_hole.y), 5, (255), 2)

        return image, centroids
