import cv2
import math

import joblib


def classify_pixel(image, classifier, pixel_x, pixel_y, w=20):
    left = pixel_x - w/2
    right = pixel_x + w/2
    top = pixel_y - w/2
    bottom = pixel_y + w/2

    patch = image[top:bottom, left:right]
    return classifier.predict(patch)


def find_edges(img):
    rows, cols = img.shape[:2]

    for i in range(rows):
        for j in range(cols):
            gray = 0.2989 * img[i, j][2] + 0.5870 * img[
                i, j][1] + 0.1140 * img[i, j][0]
            img[i, j] = [gray, gray, gray]
    img = cv2.GaussianBlur(img, (15, 15), 0)
    # cv2.imshow("Blurred", img)

    edges = cv2.Canny(img, 45, 100)
    return edges


"""
edges - black-white image produced with Canny edge detection
"""
def get_cube_corners(edges):
    rows, cols = edges.shape[:2]
    max_x_point = (-1, 0)
    max_y_point = (0, -1)
    min_x_point = (cols + 1, 0)
    min_y_point = (0, rows + 1)
    for i in range(rows):
        for j in range(cols):
            if (edges[i, j] == 0):
                continue
            else:
                if i <= min_y_point[1]:
                    min_y_point = (j, i)
                if i > max_y_point[1]:
                    max_y_point = (j, i)
                if j < min_x_point[0]:
                    min_x_point = (j, i)
                if j >= max_x_point[0]:
                    max_x_point = (j, i)

    pi = math.pi

    # theorem of cosines
    side_1 = math.dist(min_x_point, min_y_point)
    side_2 = math.dist(min_y_point, max_x_point)
    side_3 = math.dist(max_x_point, max_y_point)
    side_4 = math.dist(max_y_point, min_x_point)
    diag_1 = math.dist(min_x_point, max_x_point)
    diag_2 = math.dist(min_y_point, max_y_point)
    alpha_1 = math.acos(
        (side_1**2 + side_2**2 - diag_1**2) / (2 * side_1 * side_2))
    alpha_2 = math.acos(
        (side_2**2 + side_3**2 - diag_2**2) / (2 * side_2 * side_3))
    alpha_3 = math.acos(
        (side_3**2 + side_4**2 - diag_1**2) / (2 * side_3 * side_4))
    alpha_4 = math.acos(
        (side_4**2 + side_1**2 - diag_2**2) / (2 * side_4 * side_1))
    area = (side_1 * side_2 * math.sin(alpha_1) / 2) + (side_3 * side_4 *
                                                        math.sin(alpha_3) / 2)
    diagonals_intersection_angle = math.asin(2 * area / (diag_1 * diag_2))

    angle_treshold = 20 * pi / 180
    if (abs(alpha_1 - pi / 2)
            > angle_treshold) or (abs(alpha_2 - pi / 2) > angle_treshold) or (
                abs(alpha_3 - pi / 2) > angle_treshold) or (
                    abs(alpha_4 - pi / 2)
                    > angle_treshold) or (abs(diagonals_intersection_angle -
                                              pi / 2) > angle_treshold):
        return (min_x_point[0], min_y_point[1]), (
            max_x_point[0], min_y_point[1]), (max_x_point[0],
                                              max_y_point[1]), (min_x_point[0],
                                                                max_y_point[1])
    else:
        vertical_treshold = (max_x_point[0] + min_x_point[0]) / 2

        if min_y_point[0] < vertical_treshold:
            return min_y_point, max_x_point, max_y_point, min_x_point
        else:
            return min_x_point, min_y_point, max_x_point, max_y_point


def contoured_image(image_name):
    original = cv2.imread(image_name)
    original = cv2.resize(original, (540, 540), interpolation=cv2.INTER_LINEAR)
    img = original.copy()

    edges = find_edges(img)

    get_cube_corners(edges=edges)
    u_l, u_r, l_r, l_l = get_cube_corners(edges)

    original = cv2.line(original, u_l, u_r, (0, 255, 0), 3)
    original = cv2.line(original, u_r, l_r, (0, 255, 0), 3)
    original = cv2.line(original, l_r, l_l, (0, 255, 0), 3)
    original = cv2.line(original, l_l, u_l, (0, 255, 0), 3)
    return original, u_l, u_r, l_r, l_l


def side_colors(image, up_left, up_right, low_right, low_left, classifier):
    primary_diag = (low_right[0] - up_left[0], low_right[1] - up_left[1])
    secondary_diag = (up_left[0] - low_left[0], up_left[1] - low_left[1])
    up_side = (up_right[0] - up_left[0], up_right[1] - up_left[1])
    left_side = (low_left[0] - up_left[0], low_left[1] - up_left[1])

    # points are ordered by row and column
    p_1 = (int(up_left[0] + (1 / 6) * primary_diag[0]),
           int(up_left[1] + (1 / 6) * primary_diag[1]))
    p_9 = (int(up_left[0] + (5 / 6) * primary_diag[0]),
           int(up_left[1] + (5 / 6) * primary_diag[1]))
    p_5 = (int(up_left[0] + (1 / 2) * primary_diag[0]),
           int(up_left[1] + (1 / 2) * primary_diag[1]))
    p_7 = (int(low_left[0] + (1 / 6) * secondary_diag[0]),
           int(low_left[1] + (1 / 6) * secondary_diag[1]))
    p_3 = (int(low_left[0] + (5 / 6) * secondary_diag[0]),
           int(low_left[1] + (5 / 6) * secondary_diag[1]))
    p_2 = (int(up_left[0] + (1 / 2) * up_side[0] + (1 / 6) * left_side[0]),
           int(up_left[1] + (1 / 2) * up_side[1] + (1 / 6) * left_side[1]))
    p_8 = (int(up_left[0] + (1 / 2) * up_side[0] + (5 / 6) * left_side[0]),
           int(up_left[1] + (1 / 2) * up_side[1] + (5 / 6) * left_side[1]))
    p_4 = (int(up_left[0] + (1 / 6) * up_side[0] + (1 / 2) * left_side[0]),
           int(up_left[1] + (1 / 6) * up_side[1] + (1 / 2) * left_side[1]))
    p_6 = (int(up_left[0] + (5 / 6) * up_side[0] + (1 / 2) * left_side[0]),
           int(up_left[1] + (5 / 6) * up_side[1] + (1 / 2) * left_side[1]))

    colors = []
    colors.append(classify_pixel(image, classifier, p_1[0], p_1[1]))
    colors.append(classify_pixel(image, classifier, p_2[0], p_2[1]))
    colors.append(classify_pixel(image, classifier, p_3[0], p_3[1]))
    colors.append(classify_pixel(image, classifier, p_4[0], p_4[1]))
    colors.append(classify_pixel(image, classifier, p_5[0], p_5[1]))
    colors.append(classify_pixel(image, classifier, p_6[0], p_6[1]))
    colors.append(classify_pixel(image, classifier, p_7[0], p_7[1]))
    colors.append(classify_pixel(image, classifier, p_8[0], p_8[1]))
    colors.append(classify_pixel(image, classifier, p_9[0], p_9[1]))
    return colors


def main():
    classifier = joblib.load('color_classification_models/knn_color_classifier.sav')

    for i in range(1, 3):
        img_name = f'images/unsolved/cube_test_{i}'
        img, u_l, u_r, l_r, l_l = contoured_image(image_name=img_name)
        # small blur for better color recognition
        # img = cv2.GaussianBlur(img, (5, 5), 0)
        cv2.imshow(img_name, img)
        side_squares_colors = side_colors(img, u_l, u_r, l_r, l_l, classifier)
        print(side_squares_colors)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()