import cv2


def cluster_points(points, nclusters):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(
        points, nclusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    return centers
