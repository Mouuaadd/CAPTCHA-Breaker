import numpy as np
import matplotlib.pyplot as plt


def hough_accumulator(image, rho, theta):

    # -------------------------------------------------------- #
    # -------- 1. Definition of the accumulator array -------- #
    # -------------------------------------------------------- #

    # Get image dimensions
    # y for rows and x for columns
    Ny, Nx = image.shape

    Ntheta = int(180.0 / theta)
    Nrho = int(np.floor(np.sqrt(Nx * Nx + Ny * Ny)) / rho)

    dtheta = np.pi / Ntheta
    drho = np.floor(np.sqrt(Nx * Nx + Ny * Ny)) / Nrho

    accum = np.zeros((Ntheta, Nrho))

    # -------------------------------------------------------- #
    # -------- 2. Loop over all the pixels in the image ------- #
    # -------------------------------------------------------- #

    for y in range(Ny):
        for x in range(Nx):
            # 3. If the pixel is an edge pixel
            if image[y, x] > 0:
                # 4. Loop over all the values of theta
                for i_theta in range(Ntheta):
                    theta = i_theta * dtheta
                    rho = x * np.cos(theta) + (Ny - y) * np.sin(theta)
                    i_rho = int(rho / drho)
                    if (i_rho < Nrho) and (i_rho > 0):
                        accum[i_theta, i_rho] += 1

    return accum, Ntheta, Nrho, dtheta, drho


def houghLine(
    image, rho=1, theta=1, threshold=100.0, min_line_len=1.0, max_line_gap=1.0
):
    """Basic Hough line transform that builds the accumulator array
    Input : image : edge image (binary)
    Output : accumulator : the accumulator of hough space
             thetas : values of theta (-90 : 90)
             rs : values of radius (-max distance : max distance)
    """
    # -------------------------------------------------------- #
    # -------- 1. Instanciation of the accumulator array ----  #
    # -------------------------------------------------------- #

    accumulator, Ntheta, Nrho, dtheta, drho = hough_accumulator(image, rho, theta)

    # -------------------------------------------------------- #
    # ------------- 2. Keep values over treshold  -----------  #
    # -------------------------------------------------------- #

    accum_thresholded = accumulator.copy()
    for i_theta in range(Ntheta):
        for i_rho in range(Nrho):
            if accumulator[i_theta][i_rho] < threshold:
                accum_thresholded[i_theta][i_rho] = 0

    # -------------------------------------------------------- #
    # --------- 3. List lines polar coordinates -------------  #
    # -------------------------------------------------------- #

    lines = []
    for i_theta in range(Ntheta):
        for i_rho in range(Nrho):
            if accum_thresholded[i_theta][i_rho] != 0:
                theta = i_theta * dtheta
                rho = i_rho * drho
                lines.append((theta, rho))

    # -------------------------------------------------------- #
    # --------- 4. List lines cartesian coordinates ---------- #
    # -------------------------------------------------------- #

    lines_cartesian = []

    for theta, rho in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        lines_cartesian.append((x1, y1, x2, y2))

    # -------------------------------------------------------- #
    # -------- 5. Merge lines that are close to each other -- #
    # -------------------------------------------------------- #

    lines_cartesian_merged = []
    for line in lines_cartesian:
        x1, y1, x2, y2 = line
        merged = False
        for i in range(len(lines_cartesian_merged)):
            x1_m, y1_m, x2_m, y2_m = lines_cartesian_merged[i]
            if (
                (np.abs(x1 - x1_m) < max_line_gap)
                and (np.abs(y1 - y1_m) < max_line_gap)
                and (np.abs(x2 - x2_m) < max_line_gap)
                and (np.abs(y2 - y2_m) < max_line_gap)
            ):
                lines_cartesian_merged[i] = (
                    int((x1 + x1_m) / 2),
                    int((y1 + y1_m) / 2),
                    int((x2 + x2_m) / 2),
                    int((y2 + y2_m) / 2),
                )
                merged = True
        if not merged:
            lines_cartesian_merged.append(line)

    # -------------------------------------------------------- #
    # -------- 6. Keep lines that are long enough ------------ #
    # -------------------------------------------------------- #

    lines_cartesian_merged_filtered = []
    for line in lines_cartesian_merged:
        x1, y1, x2, y2 = line
        if np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) > min_line_len:
            lines_cartesian_merged_filtered.append(line)

    # -------------------------------------------------------- #
    # -------- 7. Plot the lines ----------------------------- #
    # -------------------------------------------------------- #

    for line in lines_cartesian_merged_filtered:
        x1, y1, x2, y2 = line
        plt.plot([x1, x2], [y1, y2], color="b")

    # -------------------------------------------------------- #
    # --- 8. Plot the image with the lines using matplotlib --- #
    # -------------------------------------------------------- #

    plt.imshow(image)
    plt.show()

    return image, lines_cartesian_merged_filtered
