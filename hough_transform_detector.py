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


class HoughTransformDetector(BaseDetector):
    """
    A class for detecting centroïds of two puzzle pieces in a CAPTCHA image using Hough transform.

    Attributes:
        None
    """

    def __init__(self):
        """
        Initializes a HoughTransformDetector object.
        """
        pass

    @staticmethod
    def _format_output(output):
        centroids = []
        for centroid in output:
            centroids.append(Centroid(centroid[0], centroid[1]))
        return centroids

    def detect(
        self, path, visualize: bool = False
    ) -> Union[List[Centroid], np.ndarray]:
        """
        Applies Hough transform to detect line segments and intersections of these lines in the image at the specified path.

        Parameters:
            path (str): The filepath of the image to be processed.

        Returns:
            list: A list of tuples representing the (x, y) coordinates of
                the detected intersections.
        """
        (
            visualization,
            centroids,
        ) = HoughTransformDetector.full_hough_transform_detection_pipe(path)

        image_number = re.search(r"([0-9]+)(\.png)", path).group(1)

        # if results folder does not exist create one
        if not os.path.exists("results/hough_transform"):
            os.makedirs("results/hough_transform")

        path_to_save = f"results/hough_transform/result_hough_img_{image_number}.png"
        if visualize:
            # save the visualization
            cv2.imwrite(path_to_save, visualization)
            return visualization

        return HoughTransformDetector._format_output(centroids)

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
    def apply_hough_transform(
        image, rho=1, theta=np.pi / 180, threshold=40, maxLineGap=50, minLineLength=10
    ):
        """
        Apply Hough transform to detect line segments in an image.

        Parameters:
            image (ndarray): The image to apply Hough transform to.
            rho (int): The distance resolution of the accumulator in pixels.
            theta (float): The angle resolution of the accumulator in radians.
            threshold (int): The accumulator threshold parameter. Only those lines are returned that get enough votes (>threshold).
            maxLineGap (int): The maximum allowed gap between two points to be considered in the same line.
            minLineLength (int): The minimum line length. Line segments shorter than that are rejected.

        Returns:
            lines: A ndarray of lines detected in the image.
        """
        # Convert to uint8
        binary_captcha = image.astype(np.uint8)

        # Apply Hough Transform
        lines = cv2.HoughLinesP(
            binary_captcha,
            rho=rho,
            theta=theta,
            threshold=threshold,
            maxLineGap=maxLineGap,
            minLineLength=minLineLength,
        )

        return lines

    @staticmethod
    def segment_lines(lines, threshold):
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                if abs(x2 - x1) < threshold:
                    vertical_lines.append(line)
                elif abs(y2 - y1) < threshold:
                    horizontal_lines.append(line)

        return horizontal_lines, vertical_lines

    @staticmethod
    def borders_avoidance(lines, threshold):
        """
        Allow to avoid the borders of the image.
        """

        new_lines = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                if (
                    x1 > threshold
                    and x2 > threshold
                    and y1 > threshold
                    and y2 > threshold
                ):
                    new_lines.append(line)
        return new_lines

    @staticmethod
    def find_intersection(line1, line2):
        # extract points
        x1, y1, x2, y2 = line1[0]
        x3, y3, x4, y4 = line2[0]

        # compute determinant
        Px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        )
        Py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        )

        return Px, Py

    def get_corners(lines, delta=5):
        horizontal_lines, vertical_lines = HoughTransformDetector.segment_lines(
            HoughTransformDetector.borders_avoidance(lines, delta), 1
        )

        # find the intersection points (corners)
        Px = []
        Py = []
        for h_line in horizontal_lines:
            for v_line in vertical_lines:
                px, py = HoughTransformDetector.find_intersection(h_line, v_line)
                Px.append(px)
                Py.append(py)

        # use clustering to find the centers of the data clusters
        corners = np.float32(np.column_stack((Px, Py)))

        return corners

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

        centroids = cluster_points(points, n_clusters)

        return centroids

    @staticmethod
    def full_hough_transform_detection_pipe(path):
        """
        Runs the full Hough transform detection pipeline.

        Parameters:
            path (str): The filepath of the image to be processed.

        Returns:
            list: A list of tuples representing the (x, y) coordinates of
                the detected intersections.
        """
        # Load the image
        image = cv2.imread(path)

        # Apply preprocessing
        binary_captcha = HoughTransformDetector.apply_preprocessing(
            image, blur=BLUR, threshold=THRESHOLD
        )

        # Apply Hough transform
        lines = HoughTransformDetector.apply_hough_transform(binary_captcha)

        # Find the corners
        corners = HoughTransformDetector.get_corners(lines)

        # Find the corners mean (corners centroids)
        corners_centroids = HoughTransformDetector.get_centroids(corners, n_clusters=8)

        # Find the centroids
        centroids = HoughTransformDetector.get_centroids(
            corners_centroids, n_clusters=2
        )

        centroid_puzzle_piece = Centroid(*centroids[0])
        centroid_hole = Centroid(*centroids[1])

        # Visualize the results
        cv2.circle(
            image,
            (int(centroid_puzzle_piece.x), int(centroid_puzzle_piece.y)),
            5,
            (255, 0, 255),
            -1,
        )
        cv2.circle(
            image, (int(centroid_hole.x), int(centroid_hole.y)), 5, (255, 0, 255), -1
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
    detector = HoughTransformDetector()

    # Detect the corners
    detector.detect(path=args.img_path, visualize=args.vizualize)

