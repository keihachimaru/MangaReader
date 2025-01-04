import cv2
import numpy as np

import cv2
import numpy as np

def detect_gaps(img):
    gray = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    blurred = gray#cv2.GaussianBlur(gray, (1, 1), 0)

    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((1, 1), np.uint8)  # 3x3 kernel for cleaning up small bumps
    straightened = cv2.dilate(thresh, kernel, iterations=1)  # Expand the lines
    straightened = cv2.erode(straightened, kernel, iterations=1)  # Shrink back, removing small bumps

    # Display the straightened image
    cv2.imshow("After Morphological Operations", straightened)
    cv2.waitKey(0)

    # Detect edges using Canny
    edges = cv2.Canny(straightened, 100, 250, apertureSize=3)

    # Display the edges
    # cv2.imshow("Edges Detected", edges)
    # cv2.waitKey(0)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, 
                            minLineLength=100, maxLineGap=5)

    # Draw detected lines on the original image
    output = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red lines
    
    # Display the final output
    # cv2.imshow("Detected Gaps with Lines", output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Load comic image
img = cv2.imread("C:/Users/Kei/Desktop/Kokou no Hito Chapter 1/0001-011.png")

# Detect and display gaps
output = detect_gaps(img)
cv2.imshow("Detected Gaps", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
