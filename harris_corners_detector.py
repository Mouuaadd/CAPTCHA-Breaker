import numpy as np
import cv2
import time
import re
import os
from typing import List, Tuple, Union
from skimage.feature import corner_peaks, corner_harris

from base_detector import BaseDetector, Centroid
from utils.squares_finder_bruteforce import find_squares
from utils.cluster_points import cluster_points

from argparse import ArgumentParser

# Preprocessing parameters
BLUR = False
THRESHOLD = 25

# Parameters for squares_finder func
SQUARE_SIZE = 75
SIZE_DELTA = 10
X_ALIGNMENT_DELTA = 10
Y_ALIGNMENT_DELTA = 10


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

    @staticmethod
    def _format_output(output):
        centroids = []
        for centroid in output:
            centroids.append(Centroid(centroid[0], centroid[1]))
        return centroids

    @staticmethod
    def is_good_n_centroids(centroids):
        return False if len(centroids) < 2 else True

    @staticmethod
    def create_fake_centroid(centroids):
        height = 318
        width = 516
        centroids = list(centroids)
        x_left = width // 4
        y = height // 2
        x_right = height // 2
        if len(centroids) == 1:
            centroids.append([x_right, y])
        else:
            centroids = []
            centroids.append([x_left, y])
            centroids.append([x_right, y])
        return centroids

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
        if not HarrisCornersDetector.is_good_n_centroids(centroids):
            centroids = HarrisCornersDetector.create_fake_centroid(centroids)

            # if results folder does not exist create one
        if not os.path.exists("results/harris_corner_detector"):
            os.makedirs("results/harris_corner_detector")

        image_number = re.search(r"([0-9]+)(\.png)", path).group(1)

        path_to_save = (
            f"results/harris_corner_detector/result_harris_img_{image_number}.png"
        )

        if visualize:
            # save the visualization
            cv2.imwrite(path_to_save, visualization)
            print(f"Vizualization of centroids saved at {path_to_save} !")

        return HarrisCornersDetector._format_output(centroids)

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
        Detects corners in an image using Harris Corners Detector.

        Parameters:
            image (ndarray): The image to apply the Harris corner detector to.
            min_distance (int): The minimum distance between two corners.

        Returns:
            ndarray: A matrix of corners coordinates in the image.
        """
        # Define the minimum distance between two corners
        min_distance = 10

        # Compute the corner response using the Harris corner detector
        corner_response = corner_harris(
            image=image, method="k", k=0.04, sigma=3)

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
    def squares_filter(
        corners,
        square_size=75,
        size_delta=10,
        x_alignment_delta=10,
        y_alignment_delta=10,
    ):
        """
        Filters out points that form squares from a list of corners coordinates.

        Parameters:
            corners (list): A list of corners.
            distance_min (int): The minimum distance between two corners.
            distance_max (int): The maximum distance between two corners.
            length_delta (int): The maximum difference between the length of two sides of a square.
            angle_delta (int): The maximum difference between the angle of two sides of a square.

        Returns:
            rectangles: A list of squares.
        """
        # Convert to list
        points = corners.tolist()

        # Find squares and rectangles in the set of points
        squares = find_squares(
            points, square_size, size_delta, x_alignment_delta, y_alignment_delta
        )

        print(f"Nb of squares found:{len(squares)}")

        return squares

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

        points = np.array(points).reshape(-1, 2).astype(np.float32)

        # Compute centroids
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
        binary = HarrisCornersDetector.apply_preprocessing(
            image, blur=BLUR, threshold=THRESHOLD
        )

        # Detect corners
        corners = HarrisCornersDetector.detect_corners(binary)

        # Filter out rectangles
        squares = HarrisCornersDetector.squares_filter(
            corners,
            square_size=SQUARE_SIZE,
            size_delta=SIZE_DELTA,
            x_alignment_delta=X_ALIGNMENT_DELTA,
            y_alignment_delta=Y_ALIGNMENT_DELTA,
        )

        # Compute centroids
        if len(squares) == 1:
            print("Only 1 square found,computing centroids of only square")
            centroids = HarrisCornersDetector.get_centroids(
                squares, n_clusters=1)

            # Get the only centroid available
            centroid = Centroid(*centroids[0])
            cv2.circle(
                image,
                (int(centroid.y), int(centroid.x)),
                3,
                (255),
                2,
            )

        else:
            centroids = HarrisCornersDetector.get_centroids(
                squares, n_clusters=2)

            # Get the centroids piecewise
            centroid_puzzle_piece = Centroid(*centroids[0])
            centroid_hole = Centroid(*centroids[1])

            # Visualize the results
            cv2.circle(
                image,
                (int(centroid_puzzle_piece.y), int(centroid_puzzle_piece.x)),
                5,
                (255, 0, 255),
                -1,
            )
            cv2.circle(
                image,
                (int(centroid_hole.y), int(centroid_hole.x)),
                5,
                (255, 0, 255),
                -1,
            )

        return image, centroids


if __name__ == "__main__":

    start_time = time.time()

    # Parse the arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--img_path",
        dest="img_path",
        default="data/0.png",
        type=str,
        help="Path to the image on which to apply detection.",
    )

    parser.add_argument(
        "--viz",
        dest="vizualize",
        default=True,
        type=bool,
        help="Vizualize result of detect or not.",
    )

    args = parser.parse_args()

    # Initialize the detector
    detector = HarrisCornersDetector()

    # Detect the corners
    detector.detect(path=args.img_path, visualize=args.vizualize)

    print(f"Total time: {time.time() - start_time}")
