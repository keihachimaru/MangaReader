import os
import cv2
import numpy as np

def detect_gaps(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Invert the image (optional, depending on your preference)
    inverted = blurred  # cv2.bitwise_not(blurred)
    
    # Apply threshold to get a binary image
    thresh = cv2.adaptiveThreshold(
        inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )


    cv2.imshow("Edges Detected", thresh)
    cv2.waitKey(0)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    panels = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 8000:
            cropped_panel = img[y:y + h, x:x + w]
            panels.append({"x": x, "y": y, "w": w, "h": h, "panel_img": cropped_panel})

    # Filter panels
    filtered_panels = []
    for panel in panels:
        x1, y1, w1, h1 = panel["x"], panel["y"], panel["w"], panel["h"]
        is_inside = False
        for other in panels:
            if panel == other:
                continue
            x2, y2, w2, h2 = other["x"], other["y"], other["w"], other["h"]
            if x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2):
                is_inside = True
                break
        if not is_inside:
            filtered_panels.append(panel)

    filtered_panels = sorted(filtered_panels, key=lambda p: (p["y"], -p["x"]))

    # Create output image with bounding rectangles
    output = img.copy()

    for panel in filtered_panels:
        x, y, w, h = panel["x"], panel["y"], panel["w"], panel["h"]
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return output

# Load the comic panel image (make sure it's already in black and white)
pages = [file for file in os.listdir("C:/Users/Kei/Desktop/Kokou no Hito Chapter 1") if file.endswith('.png')]
img = cv2.imread("C:/Users/Kei/Desktop/Kokou no Hito Chapter 1/"+pages[2])

# Detect and display gaps
output = detect_gaps(img)

# Show the result
cv2.imshow("Detected Gaps", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
