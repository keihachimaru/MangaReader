import os
import cv2
import numpy as np
from parameters import *
import random

manga = MANGAS[0]
pages = [file for file in os.listdir(manga) if file.endswith('.png')]
page = 26#random.randint(0, len(pages))
print(page)
img_path = manga+'/'+pages[page]
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

def shape_fits(img, i, stats, mask):
    x, y, width, height, area = stats[i]

    border_top = y <= PADDING * 2
    border_bottom = (y+height) >= (img.shape[0]-PADDING * 2)
    
    border_left = x <= PADDING * 2
    border_right = (x+height) >= (img.shape[1]-PADDING * 2)

    rectangle_border = (border_top and border_bottom) or (border_left and border_right)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]

    area = cv2.contourArea(contour)
    x,y, w, h = cv2.boundingRect(contour)
    perimeter = cv2.arcLength(contour, True)
    aspect_ratio = w/h if w>h else h/w

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    is_rectangle = (perimeter / (w*2+h*2)) < 1 and (perimeter / (w*2+h*2)) > .95
    is_rectangle = is_rectangle and aspect_ratio > 10 and rectangle_border
    is_squeleton = (int(border_top)+int(border_bottom)+int(border_left)+int(border_right))>1

    return ((area > 10000) and (is_rectangle or is_squeleton))

def process_img(img):
    inverted = cv2.bitwise_not(img)
    blurred = cv2.GaussianBlur(inverted, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    cv2.imshow("Filtered Shapes", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return thresh

def process_img2(img):
    inverted = cv2.bitwise_not(img)
    blurred = cv2.GaussianBlur(inverted, (1, 1), 0)
    _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    cv2.imshow("Filtered Shapes", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return thresh

def get_gaps(img):
    img[:PADDING, :] = 0
    img[-PADDING:, :] = 0
    img[:, :PADDING] = 0
    img[:, -PADDING:] = 0  

    thresh = process_img(img)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    output_image = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255#cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB).copy()

    for i in range(1, num_labels):
        mask = (labels == i).astype(np.uint8)
        x, y, width, height, area = stats[i]

        if not shape_fits(img, i, stats, mask):
            continue

        mask = (labels == i).astype(np.uint8)

        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8).tolist()
        output_image[mask == 1] = color

        contours, hierarchies = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.imshow("Filtered Shapes", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_panels(img):
    thresh = process_img2(img)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    panels = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 8000:
            cropped_panel = img[y:y + h, x:x + w]
            panels.append({"x": x, "y": y, "w": w, "h": h, "panel_img": cropped_panel})

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
    
    output_img = output_image = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255

    for panel in panels:
        x, y, w, h = panel["x"], panel["y"], panel["w"], panel["h"]
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255, 2))
        cx, cy = (x+w/2), (y+h/2)
        cv2.circle(output_image, (int(cx), int(cy)), 4, (0,0,0), -1)
        cv2.circle(output_image, (int(cx), int(cy)), 2, (0,0,255), -1)

    cv2.imshow("Filtered Shapes", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return filtered_panels

get_panels(img)