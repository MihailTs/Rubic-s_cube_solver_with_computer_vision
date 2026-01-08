from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import math
import cv2
import joblib
from PIL import Image
from torchvision import transforms
from http.server import BaseHTTPRequestHandler
import re
import json
import numpy as np
import cv2
from http.server import BaseHTTPRequestHandler

class SimpleJSONHandler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:5500")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_POST(self):
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_error(415, "Expected multipart/form-data")
            return

        # Get boundary
        boundary = content_type.split("boundary=")[1].encode()
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        # Split parts
        parts = body.split(b"--" + boundary)
        for part in parts:
            if b'Content-Disposition' in part and b'name="image"' in part:
                # Extract image bytes
                match = re.search(b'\r\n\r\n(.*)\r\n$', part, re.DOTALL)
                if not match:
                    continue
                image_bytes = match.group(1)
                break
        else:
            self.send_error(400, "Missing 'image' field")
            return

        # Decode image to NumPy array
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self.send_error(400, "Invalid image data")
            return

        # Access models from server
        classifier = self.server.classifier
        scaler = self.server.scaler
        label_mapping = self.server.label_mapping

        # Predict
        predictions = predict(image, classifier, scaler, label_mapping)

        # Send JSON response
        response_json = json.dumps({"predictions": predictions, "status": "ok"})
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:5500")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_json)))
        self.end_headers()
        self.wfile.write(response_json.encode("utf-8"))

def run(server_class=HTTPServer, handler_class=SimpleJSONHandler, port=8000, classifier=None, scaler=None, label_mapping=None):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)

    httpd.classifier = classifier
    httpd.scaler = scaler
    httpd.label_mapping = label_mapping

    print(f"Server running on http://localhost:{port}")
    httpd.serve_forever()


def classify_pixel(image, classifier, scaler, label_mapping, pixel_x, pixel_y, w=20):
    left = int(pixel_x - w/2)
    right = int(pixel_x + w/2)
    top = int(pixel_y - w/2)
    bottom = int(pixel_y + w/2)

    patch = image[top:bottom, left:right]
    
    patch_pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    train_transforms = transforms.Compose(
        [transforms.Resize((20, 20)),
         transforms.ToTensor()])
    patch_tensor = train_transforms(patch_pil)
    patch_flat = patch_tensor.view(1, -1).numpy()
    
    patch_scaled = scaler.transform(patch_flat)
    prediction_idx = classifier.predict(patch_scaled)[0]
    return label_mapping[prediction_idx]

def find_edges(img):
    rows, cols = img.shape[:2]

    for i in range(rows):
        for j in range(cols):
            gray = 0.2989 * img[i, j][2] + 0.5870 * img[
                i, j][1] + 0.1140 * img[i, j][0]
            img[i, j] = [gray, gray, gray]
    img = cv2.GaussianBlur(img, (15, 15), 0)
    # cv2.imshow("Blurred", img)

    edges = cv2.Canny(img, 55, 150)
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

def contour_image(image):
    image_resized = cv2.resize(image, (540, 540), interpolation=cv2.INTER_LINEAR)
    img = image_resized.copy()

    edges = find_edges(img)

    get_cube_corners(edges=edges)
    u_l, u_r, l_r, l_l = get_cube_corners(edges)
 
    # image_resized = cv2.line(image_resized, u_l, u_r, (0, 255, 0), 3)
    # image_resized = cv2.line(image_resized, u_r, l_r, (0, 255, 0), 3)
    # image_resized = cv2.line(image_resized, l_r, l_l, (0, 255, 0), 3)
    # image_resized = cv2.line(image_resized, l_l, u_l, (0, 255, 0), 3)
    return image_resized, u_l, u_r, l_r, l_l

def side_colors(image, up_left, up_right, low_right, low_left, classifier, scaler, label_mapping):
    primary_diag = (low_right[0] - up_left[0], low_right[1] - up_left[1])
    secondary_diag = (up_right[0] - low_left[0], up_right[1] - low_left[1])
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

    points = [p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8, p_9]
    
    colors = []
    for i, point in enumerate(points):
        colors.append(classify_pixel(image, classifier, scaler, label_mapping, point[0], point[1]))
    
    return colors

def predict(image, classifier, scaler, label_mapping):
    img, u_l, u_r, l_r, l_l = contour_image(image)
    side_squares_colors = side_colors(img, u_l, u_r, l_r, l_l, classifier, scaler, label_mapping)
    return side_squares_colors

def main():
    classifier = joblib.load('color_classification_models/knn_color_predictor.sav')
    scaler = joblib.load('color_classification_models/scaler.sav')
    label_mapping = joblib.load('color_classification_models/label_mapping.sav')
    
    run(handler_class=SimpleJSONHandler,
        classifier=classifier,
        scaler=scaler,
        label_mapping=label_mapping)


if __name__ == "__main__":
    main()