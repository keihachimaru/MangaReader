import os
import cv2
import numpy as np

PADDING = 5

def detect_and_color_largest_shapes(img_path):
    # Load the image
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

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)

    # Extract areas (column 4 in stats gives the area of each component)
    areas = stats[:, cv2.CC_STAT_AREA]

    # Ignore the background (label 0)
    areas[0] = 0  # Set background area to 0

    # Get the indices of the top 10 largest components by area
    largest_indices = np.argsort(areas)[-10:]  # Last 10 largest areas

    # Create an output image to visualize results
    output = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255# cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Assign random colors to the top 10 components
    for i, idx in enumerate(largest_indices):
        if idx == 0:
            continue  # Skip the background

        # Create a mask for the current component
        component_mask = (labels == idx).astype(np.uint8) * 255

        # Generate a random color
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8).tolist()

        # Color the component on the output image
        output[component_mask == 255] = color

        # Optional: Add the area as text
        #x, y, w, h, area = stats[idx]
        #cv2.putText(output, f"{area}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show the result
    cv2.imshow("Top 10 Largest Connected Shapes", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
pages = [file for file in os.listdir("C:/Users/Kei/Desktop/Kokou no Hito Chapter 1") if file.endswith('.png')]

for i in range(26, 27):
    detect_and_color_largest_shapes("C:/Users/Kei/Desktop/Kokou no Hito Chapter 1/"+pages[i])

