from math import atan2, pi


def angle(p1, p2):
    return atan2(p2.y - p1.y, p2.x - p1.x) * 180 / pi


def check_angle(p1, p2, angle_delta):
    if (
        ((0 - angle_delta) <= abs(angle(p1, p2)) <= (0 + angle_delta))
        or ((45 - angle_delta) <= abs(angle(p1, p2)) <= (45 + angle_delta))
        or ((90 - angle_delta) <= abs(angle(p1, p2)) <= (90 + angle_delta))
        or ((180 - angle_delta) <= abs(angle(p1, p2)) <= (180 + angle_delta))
    ):
        return True
    else:
        return False
