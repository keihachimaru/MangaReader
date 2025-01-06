import os
import cv2
import numpy as np

PADDING = 5

def filterShape(stats, img_shape):
    # Extract the bounding box (x, y, width, height, area)
    x, y, w, h, area = stats

    # 1. Check for very thin rectangles (aspect ratio check)
    aspect_ratio = w / h if h > w else h / w  # Aspect ratio of the bounding box

    if aspect_ratio > 5:  # Arbitrary threshold for thin shapes (adjust as needed)
        return False

    # 2. Check if the area is too small (e.g., it's a small gap or noise)
    if area < 500:  # Arbitrary minimum area for gaps, you may adjust this value
        return False

    # 3. Check if the shape is close to the image borders (likely part of a gap)
    img_height, img_width = img_shape
    border_threshold = 20  # Distance from the image border to be considered close

    if x < border_threshold or y < border_threshold or x + w > img_width - border_threshold or y + h > img_height - border_threshold:
        return False

    # 4. Convex hull check: We expect gaps to have relatively simple convex hulls
    contour = np.array([[(x, y), (x+w, y), (x+w, y+h), (x, y+h)]], dtype=np.int32)  # Approximation of a rectangle
    if cv2.isContourConvex(contour):
        return False

    # 5. Optionally, check if the contour is a polygon (approximation of a rectangle)
    # We use approxPolyDP to approximate a polygon with fewer vertices
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:  # Likely a rectangle (with 4 corners)
        return False

    # If none of the conditions were met, return True (meaning it's a valid shape)
    return True

def display_result(shapes, img, labels):
    output = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255

    for i, idx in enumerate(shapes):
        if idx == 0:
            continue

        component_mask = (labels == idx).astype(np.uint8) * 255
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8).tolist()
        output[component_mask == 255] = color

    cv2.imshow("Filtered Shapes", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Simple filter function template, always returns True (no shape filtering)
def filterShape(stats):
    return True  # Keep all shapes

def detect_and_color_largest_shapes(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    img[:PADDING, :] = 0
    img[-PADDING:, :] = 0
    img[:, :PADDING] = 0
    img[:, -PADDING:] = 0  

    inverted = cv2.bitwise_not(img)
    blurred = cv2.GaussianBlur(inverted, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    filtered_shapes = [idx for idx in range(1, num_labels) if True]

    output_image = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255

    for i in filtered_shapes:
        mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            cv2.drawContours(output_image, [contour], -1, (0, 0, 255), 2)  # Draw in green

    cv2.imshow("Filtered Shapes", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
pages = [file for file in os.listdir("C:/Users/Kei/Desktop/Kokou no Hito Chapter 1") if file.endswith('.png')]


for i in range(0, 1):
    detect_and_color_largest_shapes("C:/Users/Kei/Desktop/Kokou no Hito Chapter 1/" + pages[i])
