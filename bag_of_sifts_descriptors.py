import numpy as np
import cv2
import os
from typing import List, Tuple, Union

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances

# Random Forest Regressor@
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from skimage.io import imread
import pickle

from base_detector import BaseDetector, Centroid
from utils.clusterPoints import cluster_points
from utils.loadJsonData import load_json_data


class BagOfSIFTSDetector(BaseDetector):
    """
    A class for that extracts SIFT descriptors from an image, clusters them using K-means and then uses the resulting to apply machine learning regression algorithms.

    Attributes:
        None
    """

    def __init__(self):
        """
        Initializes a BagOfSIFTSDetector object.
        """
        pass

    def detect(
        self, path, visualize: bool = False
    ) -> Union[List[Centroid], np.ndarray]:
        """
        Uses bag of SIFT descriptors as features for Machine Learning regression tasks.

        Parameters:
            path (str): The filepath of the image to be processed.

        Returns:
        """
        # Check if model exists
        filename = "model.pkl"
        if not os.path.exists(filename):
            print("No 'model.pkl' file found")
            print("Training model before detection...")
            self.train()
        else:
            print("Model found!")

            # load the model from disk
            loaded_model = pickle.load(open(filename, "rb"))
            centroids = loaded_model.predict()

        (
            visualization,
            centroids,
        ) = BagOfSIFTSDetector.full_bag_of_sifts_detection_pipe(path)

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
    # load the json data
    def get_image_paths(data_path, train_size=0.7):
        total_images_paths = []

        train_image_paths = []
        test_image_paths = []
        train_labels = []
        test_labels = []

        for file in os.listdir(data_path):
            if file.endswith(".json"):
                image_path, target_puzzle_piece, target_hole = load_json_data(
                    os.path.join(data_path, file)
                )
                total_images_paths.append(
                    [image_path, target_puzzle_piece, target_hole]
                )

        # Shuffle the numpy array data
        np.random.shuffle(total_images_paths)

        # Split the data into train and test
        train_size = int(len(total_images_paths) * train_size)
        train_data = total_images_paths[:train_size]
        test_data = total_images_paths[train_size:]

        # Retrieve train_image_paths, test_image_paths, train_labels, test_labels
        for data in train_data:
            train_image_paths.append(os.path.join(data_path, data[0]))
            train_labels.append(data[1:])

        for data in test_data:
            test_image_paths.append(os.path.join(data_path, data[0]))
            test_labels.append(data[1:])

        # Convert to numpy array
        train_image_paths = np.array(train_image_paths)
        test_image_paths = np.array(test_image_paths)
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        # Ravel train and test labels
        train_labels = np.array([x.flatten() for x in train_labels])
        test_labels = np.array([x.flatten() for x in test_labels])

        return train_image_paths, test_image_paths, train_labels, test_labels

    def get_corners(self, lines, delta=5):
        horizontal_lines, vertical_lines = self.segment_lines(
            self.borders_avoidance(lines, delta), 1
        )

        # find the intersection points (corners)
        Px = []
        Py = []
        for h_line in horizontal_lines:
            for v_line in vertical_lines:
                px, py = self.find_intersection(h_line, v_line)
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

        # Convert to numpy array
        Px = np.array([p.x for p in points])
        Py = np.array([p.y for p in points])
        points = np.float32(np.column_stack((Px, Py)))

        centroids = cluster_points(points, n_clusters)

        return centroids

    @staticmethod
    def build_vocabulary(image_paths, vocab_size=200):
        """
        This function will sample SIFT descriptors from the training images,
        cluster them with kmeans, and then return the cluster centers.

        Args:
        -   image_paths: list of image paths.
        -   vocab_size: size of vocabulary (cluster centers)

        Returns:
        -   vocab: This is a vocab_size x d numpy array (vocabulary). Each row is a
        cluster center / visual word
        """

        # initialise SIFT object
        sift = cv2.SIFT_create()

        features = []
        keypoints = []

        # 1. Compute SIFT
        for i, image_path in enumerate(image_paths):
            # Read the image
            img = imread(image_path)

            # Detect keypoints in the image and compute the feature descriptors for the keypoints
            keys, descriptors = sift.detectAndCompute(img, None)
            features.append(descriptors)
            keypoints.append(keys)

        # 2. Cluster using KMeans
        features = np.vstack(features)
        kmeans = MiniBatchKMeans(n_clusters=vocab_size, random_state=42)
        kmeans.fit(features)

        vocab = kmeans.cluster_centers_

        return vocab

    @staticmethod
    def get_bags_of_sifts(self, image_paths, vocab_filename, vocab_size=200):
        """
        You will want to construct SIFT features here in the same way you
        did in build_vocabulary() and then assign each local feature to its nearest
        cluster center and build a histogram indicating how many times each cluster was used.
        Don't forget to normalize the histogram, or else a larger image with more
        SIFT features will look very different from a smaller version of the same
        image.

        Args:
        -   image_paths: paths to N images
        -   vocab_filename: Path to the precomputed vocabulary.
            This function assumes that vocab_filename exists and contains an
            vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
            or visual word. This ndarray is saved to disk rather than passed in
            as a parameter to avoid recomputing the vocabulary every run.

        Returns:
        -   features: N x d matrix, where d is the dimensionality of the
            feature representation. In this case, d will equal the number of
            clusters or equivalently the number of entries in each image's
            histogram (vocab_size) below.
        """

        # initialise SIFT object
        sift = cv2.SIFT_create()

        # load vocabulary
        with open(vocab_filename, "rb") as f:
            vocab = pickle.load(f)

        features = []

        for i, image_path in enumerate(image_paths):
            # Read the image
            img = imread(image_path)

            # Apply preprocessing

            # Detect keypoints in the image and compute the feature descriptors for the keypoints
            _, descriptors = sift.detectAndCompute(img, None)

            # Compute pairwise distances between each cluster and descriptor vector
            dists = pairwise_distances(vocab, descriptors)

            # Take arg that minimizes the distance
            argmin = np.argmin(dists, 0)

            # Histogram with a fixed size of 200 words
            hist = np.bincount(argmin, minlength=vocab_size)

            # Normalize
            hist = hist / hist.sum()

            features.append(hist)

        return np.vstack(features)

    @staticmethod
    def pred_bag_of_sifts_detection_pipe(path):
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
        binary_captcha = BagOfSIFTSDetector.apply_preprocessing(image, blur=True)

        # Apply Hough transform
        lines = BagOfSIFTSDetector.apply_hough_transform(binary_captcha)

        # Find the corners
        corners = BagOfSIFTSDetector.get_corners(lines)

        # Find the corners mean (corners centroids)
        corners_centroids = BagOfSIFTSDetector.get_centroids(corners, n_clusters=8)

        # Find the centroids
        centroids = BagOfSIFTSDetector.get_centroids(corners_centroids, n_clusters=2)

        centroid_puzzle_piece = Centroid(*centroids[0])
        centroid_hole = Centroid(*centroids[1])

        # Visualize the results
        cv2.circle(
            image, (centroid_puzzle_piece.x, centroid_puzzle_piece.y), 5, (255), 2
        )
        cv2.circle(image, (centroid_hole.x, centroid_hole.y), 5, (255), 2)

        return image, centroids

    def train(self, train_size=0.7, vocab_size=200):

        print("-" * 100)
        print("Start of model training...")
        data_path = os.path.join(".", "data")
        (
            train_image_paths,
            test_image_paths,
            train_labels,
            test_labels,
        ) = self.get_image_paths(data_path, train_size=train_size)

        print(f"Shape of training set: {train_image_paths.shape}")
        print(f"Shape of test set: {test_image_paths.shape}")
        print(f"Shape of training labels: {train_labels.shape}")
        print(f"Shape of test labels: {test_labels.shape}")

        # initialise SIFT object
        sift = cv2.SIFT_create()

        vocab_filename = "vocab.pkl"
        vocab_size = vocab_size  # Larger values :=> slower to compute

        print("-" * 10)
        print("Building vocabulary...")
        vocab = self.build_vocabulary(train_image_paths, vocab_size)
        with open(vocab_filename, "wb") as f:
            pickle.dump(vocab, f)
            print("{:s} saved".format(vocab_filename))

        print("-" * 10)
        print("Building train & test features...")
        train_image_feats = self.get_bags_of_sifts(
            train_image_paths, vocab_filename, vocab_size
        )
        test_image_feats = self.get_bags_of_sifts(
            test_image_paths, vocab_filename, vocab_size
        )

        print("Train features shape: ", train_image_feats.shape)
        print("Test features shape: ", test_image_feats.shape)

        # Initialize the regressor
        rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

        print("-" * 100)
        print("Fitting model...")
        # Fit the regressor to the data
        rf.fit(train_image_feats, train_labels)
        print("End of fitting model.")

        # Predict on the test data: y_pred
        test_pred_labels = rf.predict(test_image_feats)
        train_pred_labels = rf.predict(train_image_feats)
        print("-" * 100)

        # Compute metrics
        mse_test = mean_squared_error(test_labels, test_pred_labels)
        mse_train = mean_squared_error(train_labels, train_pred_labels)

        rmse_train = mean_squared_error(train_labels, train_pred_labels, squared=False)
        rmse_test = mean_squared_error(test_labels, test_pred_labels, squared=False)

        print(f"MSE on Train data: {mse_train} | RMSE on Train data: {rmse_train}")
        print(f"MSE on Test data: {mse_test} | RMSE on Test data: {rmse_test}")

        # save the model to disk
        filename = "model.pkl"
        pickle.dump(rf, open(filename, "wb"))

        print("Model saved successfully to disk!")

        return rf
