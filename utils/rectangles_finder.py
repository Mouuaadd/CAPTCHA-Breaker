from utils.angleCompute import check_angle
from utils.distanceCompute import distance


class Point:
    """
    Class that represents a point in the plane
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y


class QuadTree:
    """
    Class that represents a tree data structure for efficiently querying subgroups of points.
    """

    def __init__(self, points, x_min, y_min, x_max, y_max):
        self.points = points
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.nw = None
        self.ne = None
        self.sw = None
        self.se = None

    def split(self):
        x_mid = (self.x_min + self.x_max) / 2
        y_mid = (self.y_min + self.y_max) / 2
        points_nw = [p for p in self.points if p.x < x_mid and p.y < y_mid]
        points_ne = [p for p in self.points if p.x >= x_mid and p.y < y_mid]
        points_sw = [p for p in self.points if p.x < x_mid and p.y >= y_mid]
        points_se = [p for p in self.points if p.x >= x_mid and p.y >= y_mid]
        self.nw = QuadTree(points_nw, self.x_min, self.y_min, x_mid, y_mid)
        self.ne = QuadTree(points_ne, x_mid, self.y_min, self.x_max, y_mid)
        self.sw = QuadTree(points_sw, self.x_min, y_mid, x_mid, self.y_max)
        self.se = QuadTree(points_se, x_mid, y_mid, self.x_max, self.y_max)

    def query_region(self, x_min, y_min, x_max, y_max):
        if (
            x_min > self.x_max
            or x_max < self.x_min
            or y_min > self.y_max
            or y_max < self.y_min
        ):
            return []
        elif self.nw is None:
            return self.points
        else:
            points = []
            points += self.nw.query_region(x_min, y_min, x_max, y_max)
            points += self.ne.query_region(x_min, y_min, x_max, y_max)
            points += self.sw.query_region(x_min, y_min, x_max, y_max)
            points += self.se.query_region(x_min, y_min, x_max, y_max)
            return points


def find_squares_and_rectangles(
    points, distance_min, distance_max, length_delta, angle_delta
):
    """
    Main function for finding squares and rectangles in a set of points.
    Args:
        - points: list of Point objects
        - distance_max: maximum distance between two points to be considered part of the same shape
        - length_delta: maximum difference in length between points to be considered part of the same shape
        - angle_delta: maximum difference in angle of intersection between two points to be considered part of the same shape

    Returns:
        - tuple: (list(identified squares), list(identified rectangles))
    """
    # Build quadtree
    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    quadtree = QuadTree(points, x_min, y_min, x_max, y_max)
    quadtree.split()

    rectangles = []

    # Find rectangles
    for p in points:
        # Find other points within epsilon of p
        region_x_min = p.x - distance_max
        region_y_min = p.y - distance_max
        region_x_max = p.x + distance_max
        region_y_max = p.y + distance_max
        suitable_points = quadtree.query_region(
            region_x_min, region_y_min, region_x_max, region_y_max
        )
        suitable_points = [
            q
            for q in suitable_points
            if q is not p and distance_min <= distance(p, q) <= distance_max
        ]

        # Check if any triplets of suitable points form a rectangle
        for i in range(len(suitable_points)):
            for j in range(i + 1, len(suitable_points)):
                for k in range(j + 1, len(suitable_points)):
                    q1 = suitable_points[i]
                    q2 = suitable_points[j]
                    q3 = suitable_points[k]
                    if (
                        check_angle(p, q1, angle_delta)
                        and check_angle(p, q2, angle_delta)
                        and check_angle(p, q3, angle_delta)
                        and check_angle(q1, q2, angle_delta)
                        and check_angle(q1, q3, angle_delta)
                        and check_angle(q2, q3, angle_delta)
                    ):
                        print(
                            f"p.x:{p.x},p.y:{p.y} || q1.x:{q1.x},q1.y:{q1.y} || q2.x:{q2.x},q2.y:{q2.y} || q3.x:{q3.x},q3.y:{q3.y} || abs(angle(p, q1):{abs(angle(p, q1))} || abs(angle(p, q2):{abs(angle(p, q2))} || abs(angle(p, q3):{abs(angle(p, q3))}"
                        )
                        rectangles.append([p, q1, q2, q3])

    return rectangles
