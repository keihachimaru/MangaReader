import os
import cv2
import numpy as np

PADDING = 5

def display_result(shapes, img, labels):
    # Create an output image to visualize results
    output = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255# cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Assign random colors to the top 10 components
    for i, idx in enumerate(shapes):
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

def isHollow(contours, hierarchy, min_area, min_hull_area=None):
    for idx, h in enumerate(hierarchy):
        parent = h[3]  # Index of the parent contour
        first_child = h[2]  # Index of the first child contour

        # Only consider top-level parent contours (no parent: parent == -1)
        if parent == -1 and first_child != -1:
            # Calculate the hull area of the parent contour
            hull = cv2.convexHull(contours[idx])
            hull_area = cv2.contourArea(hull)

            # Check if the hull area exceeds the threshold (if specified)
            if min_hull_area is not None and hull_area < min_hull_area:
                continue  # Skip this contour if the hull area is too small

            # Check if the parent contour has a child with a significant area
            child = first_child
            while child != -1:
                child_area = cv2.contourArea(contours[child])
                if child_area > min_area:
                    return True, idx  # Return the parent contour index
                child = hierarchy[child][0]  # Move to the next child

    return False, None

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
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    # Extract areas (column 4 in stats gives the area of each component)
    areas = stats[:, cv2.CC_STAT_AREA]

    # Ignore the background (label 0)
    # areas[0] = 0  # Set background area to 0
    hollow_shapes = []
    MIN_AREA = 20000  # Set the minimum area for a hole to be considered significant

    output_image = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255

    for i in range(1, num_labels):  # Skip background (label 0)
        # Create a mask for the current connected component
        mask = (labels == i).astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0 and hierarchy is not None:
            hierarchy = hierarchy[0]  # Flatten the hierarchy

            # Check if the contour is hollow with significant child area and hull area
            min_hull_area = 20000  # Example threshold for hull area
            is_hollow, parent_idx = isHollow(contours, hierarchy, MIN_AREA, min_hull_area)

            if is_hollow:
                # Draw the parent contour in blue
                cv2.drawContours(output_image, contours, parent_idx, (255, 0, 0), 2)  # Blue

                # Draw all its child contours in red
                child = hierarchy[parent_idx][2]
                while child != -1:
                    cv2.drawContours(output_image, contours, child, (0, 0, 255), 2)  # Red
                    child = hierarchy[child][0]  # Move to the next child

                hollow_shapes.append(i)

    cv2.imshow("Top 10 Largest Connected Shapes", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #display_result(hollow_shapes, img, labels)

# Example usage
pages = [file for file in os.listdir("C:/Users/Kei/Desktop/Kokou no Hito Chapter 1") if file.endswith('.png')]

for i in range(20,30):
    detect_and_color_largest_shapes("C:/Users/Kei/Desktop/Kokou no Hito Chapter 1/"+pages[i])

