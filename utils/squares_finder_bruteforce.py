from itertools import combinations

### Simple square search (computationnaly P hard) ###


def is_square(
    points, square_side=1, size_delta=10, x_alignment_delta=10, y_alignment_delta=10
):
    """
    Checks if a set of 4 points form a square of a given size with a given delta tolerance.
    This algorithm is efficient but computationnaly expensive (P-hard).

    Parameters:
        points (list): A list of points.
        square_side (int): The size of the square.
        delta (int): The tolerance for the size of the square.
        x_alignment_delta (int): The tolerance for the alignment of the points on the x axis.
        y_alignment_delta (int): The tolerance for the alignment of the points on the y axis.

    Returns:
        bool: True if the points form a square, False otherwise.
    """
    if len(set(tuple(pt) for pt in points)) != 4:  # Some points are identical
        return False
    x_coords = sorted(set(pt[0] for pt in points))
    y_coords = sorted(set(pt[1] for pt in points))

    # Get rid of very X values near values
    x_coords = [
        x
        for i, x in enumerate(x_coords)
        if i == 0 or abs(x - x_coords[i - 1]) >= x_alignment_delta
    ]
    y_coords = [
        x
        for i, x in enumerate(y_coords)
        if i == 0 or abs(x - y_coords[i - 1]) >= y_alignment_delta
    ]

    if not (
        len(x_coords) == len(y_coords) == 2
    ):  # Points are not aligned 2 by 2 on x and on y
        return False

    if not (
        abs(x_coords[1] - x_coords[0] - square_side) <= size_delta
        and abs(y_coords[1] - y_coords[0] - square_side) <= size_delta
    ):
        # Not a square, or not the right size with the given delta tolerance
        return False
    # We have a square
    return True


def find_squares(
    pts_list, square_side=1, size_delta=10, x_alignment_delta=10, y_alignment_delta=10
):
    squares = []
    for pts in combinations(pts_list, 4):
        if is_square(
            pts, square_side, size_delta, x_alignment_delta, y_alignment_delta
        ):  # Retrieve all the 4 points that form the square
            squares.append(pts)
    return squares
