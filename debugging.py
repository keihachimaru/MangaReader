import cv2
import numpy as np

def display_contours(shapes, img, labels):
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
