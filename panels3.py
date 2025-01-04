import os
import cv2
import numpy as np

def detect_gaps(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if not already
    filtered = cv2.bilateralFilter(gray, 9, 5, 5)
    blurred = cv2.GaussianBlur(filtered, (3, 3), 0)
    inverted = cv2.bitwise_not(blurred)
    
    _, thresh = cv2.threshold(inverted, 100, 255, cv2.THRESH_BINARY)


    # Detect edges using Canny edge detection
    edges = cv2.Canny(thresh, 100, 150, apertureSize=3)

    # Apply morphological operation to close the edge
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1) 

    cv2.imshow("Edges Detected", edges)
    cv2.waitKey(0)
    
    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, 
                            minLineLength=50, maxLineGap=4)

    # Copy the original image to draw detected lines
    output = np.ones_like(img) * 255
    
    # Draw detected lines on the image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            horizontal = abs(y1-y2)<5
            vertical = abs(x1-x2)<5
            if horizontal:
                cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 1)
            elif vertical:
                cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 1)
        
    return output

# Load the comic panel image (make sure it's already in black and white)
pages = [file for file in os.listdir("C:/Users/Kei/Desktop/Kokou no Hito Chapter 1") if file.endswith('.png')]

for i in range(2,3):
    img = cv2.imread("C:/Users/Kei/Desktop/Kokou no Hito Chapter 1/"+pages[i])
    # Detect and display gaps
    output = detect_gaps(img)

    # Show the result
    cv2.imshow("Detected Gaps", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
