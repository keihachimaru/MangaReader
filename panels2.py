import cv2
import numpy as np

def detect_gaps(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 250, 255, cv2.THRESH_BINARY)
    kernel = np.ones((1, 1), np.uint8)
    straightened = cv2.dilate(thresh, kernel, iterations=1)
    straightened = cv2.erode(straightened, kernel, iterations=1)

    cv2.imshow("After Morphological Operations", straightened)
    cv2.waitKey(0)

    # Ensure the image is single-channel (grayscale)
    contours, _ = cv2.findContours(straightened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    filtered_panels = []
    for contour in contours:
        x1, y1, w1, h1 = cv2.boundingRect(contour)  # Get bounding box

        is_inside = False
        for other in contours:
            if contour is other:
                continue
            x2, y2, w2, h2 = cv2.boundingRect(other)  # Get bounding box of the other contour
            if x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2):
                is_inside = False
                break
        if not is_inside:
            filtered_panels.append((x1, y1, w1, h1))  # Add bounding box as a tuple

    # Sort filtered panels based on the y-coordinate, then x-coordinate in descending order
    filtered_panels = sorted(filtered_panels, key=lambda p: (p[1], -p[0]))

    # Create a black output image to draw the contours
    output = np.ones_like(img) * 255  # White background

    for (x, y, w, h) in filtered_panels:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle for contours

    # Display the final output with detected gaps
    cv2.imshow("Detected Gaps", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread("C:/Users/Kei/Desktop/Kokou no Hito Chapter 1/0001-011.png")

# Detect and display gaps
detect_gaps(img)
