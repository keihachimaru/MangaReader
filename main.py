import os
from moviepy.editor import *
import requests
import cv2
from PIL import Image
import json
import numpy as np
import math

###############################################################
###                     PANEL EXTRACTION                    ###
###############################################################

mangas = [
    "C:/Users/Kei/Desktop/Kokou no Hito Chapter 1",
    "C:/Users/Kei/Desktop/Dragon Ball Chapter 123",
    "C:/Users/Kei/Desktop/Dragon Ball Chapter 241",
    "C:/Users/Kei/Desktop/Frieren - Beyond Journey's End Chapter 1"
]

MANGA_PAGES_DIR = mangas[0]
MANGA_PANELS_DIR = "panels"
VIDEO_PAGES_DIR = "pages"
RESULTS_DIR = "videos"
API_KEY_FILE = "./key/api_key2.txt"
OCR_API_URL = "https://api.ocr.space/parse/image"
GET_DATA = False
PARSE = True

def preprocess_page(path):
    """
    Preprocess the input black-and-white manga page to prepare for panel detection.
    Adds a strong Gaussian blur to reduce noise before thresholding.
    Applies distance transform and watershed segmentation to handle overlapping panels.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (1, 1), 0)
    inverted = blurred#cv2.bitwise_not(blurred)
    _, thresh = cv2.threshold(inverted, 245, 255, cv2.THRESH_BINARY)

    # Step 1: Morphological operations to clean the thresholded image
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # # Step 2: Distance transform
    # dist_transform = cv2.distanceTransform(cleaned_thresh, cv2.DIST_L2, 5)
    # _, markers = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    # markers = np.uint8(markers)

    # # Step 3: Watershed segmentation
    # _, connected_components = cv2.connectedComponents(markers)
    # markers = connected_components + 1
    # markers[cleaned_thresh == 0] = 0
    # img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert grayscale image to color for watershed
    # cv2.watershed(img_color, markers)

    # Visualization for debugging
    if not GET_DATA:
        cv2.imshow("Cleaned Threshold", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img, thresh

def get_panels(thresh, img):
    """
    Detect vignettes (panels) using contours and sort them in reading order.
    If more than 8 panels are detected, return the whole image as one panel.
    """
    
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
    
    return filtered_panels

def get_long_lines(thresh, img, min_line_length=200, max_line_gap=9):
    """
    Detect and filter lines using Hough Line Transformation.
    Keep only the longer lines and return their coordinates along with the image segments.
    
    Parameters:
    - thresh: Binary thresholded image for detecting edges.
    - img: Original image for extracting line segments.
    - min_line_length: Minimum length of a line to be considered.
    - max_line_gap: Maximum allowed gap between line segments to be connected.
    
    Returns:
    - filtered_lines: List of dictionaries containing line coordinates and cropped images.
    """
    
    # Detect lines using Hough Line Transformation
    lines = cv2.HoughLinesP(
        thresh, rho=1, theta=np.pi/180, threshold=100,
        minLineLength=min_line_length, maxLineGap=max_line_gap
    )
    
    if lines is None:
        return []  # No lines detected
    
    # Prepare a list for storing valid lines and their lengths
    valid_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length >= min_line_length:
            valid_lines.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "length": length})
    
    # Sort lines by their length (longest first)
    valid_lines = sorted(valid_lines, key=lambda l: -l["length"])
    
    # Filter lines: Optionally, you can apply additional filtering based on overlap, angles, etc.
    
    filtered_lines = []
    for line in valid_lines:
        x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
        line_img = img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
        filtered_lines.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "length": line["length"],
            "line_img": line_img
        })
    
    return filtered_lines

def clean_text(text):
    """Clean and formalize the text."""
    return text

def get_text(img_path, api_key):
    with open(img_path, "rb") as f:
        response = requests.post(
            OCR_API_URL,
            files={
                "file": f
            },
            data={
                "apikey": api_key, 
                "language": "eng",
                "scale": True,
                "OCREngine": 2,
                "isOverlayRequired": True
            }
        )
    
    result = response.json()
    try:
        return result
    except:
        return ''

def retrieve_text(panels):
    api_key = load_api_key()

    for i in range(len(panels)):
        text = get_text(panels[i]["image_path"], api_key)
        panels[i]["text"] = text

def save_panels(panels, num):
    """
    Extract text from each detected panel using pytesseract and save each panel as a PNG file.
    """
    extracted_data = []
    output_dir = os.path.join(MANGA_PANELS_DIR, num)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, panel in enumerate(panels):
        panel_filename = os.path.join(output_dir, f"panel_{i}.png")
        cv2.imwrite(panel_filename, panel["panel_img"])

        extracted_data.append({
            "index": i,
            "coordinates": (panel["x"], panel["y"], panel["w"], panel["h"]),
            "image_path": panel_filename,
            "text" : None,
        })
    
    return extracted_data

def load_api_key():
    """Load the API key from a secure file."""
    try:
        with open(API_KEY_FILE, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"API key file not found at {API_KEY_FILE}")
    except Exception as e:
        raise Exception(f"Error reading API key file: {e}")

def process_manga_page(image_path, num):
    """
    Process the manga page to detect panels, sort them, and extract text from each.
    """

    original_img, thresh = preprocess_page(image_path)
    panels_contours = get_long_lines(thresh,original_img)
    #panels_contours = get_panels(thresh, original_img)
    if GET_DATA == False:
        #display_detected_panels(original_img, panels_contours)
        display_detected_lines(original_img, panels_contours)
    panels = save_panels(panels_contours, num)

    if GET_DATA:
        retrieve_text(panels)

    return panels

###############################################################
###                      VIDEO CREATION                     ###
###############################################################

TARGET_ASPECT_RATIO = 9 / 16
VIDEO_WIDTH = 720
VIDEO_HEIGHT = int(VIDEO_WIDTH / TARGET_ASPECT_RATIO)
BASE_DURATION = 1
MOVEMENT_SPEED = 600
READ_SPEED = 3

def preprocess_image(image_path):
    """Convert the image to RGB and resize it to fit one of the dimensions (width or height) of the target aspect ratio."""
    formatter = {"PNG": "RGBA", "JPEG": "RGB"}
    img = Image.open(image_path)
    
    rgbimg = Image.new(formatter.get(img.format, 'RGB'), img.size)
    rgbimg.paste(img)

    img_width, img_height = img.size
    if (img_width / img_height) < TARGET_ASPECT_RATIO:
        scale = VIDEO_WIDTH / img_width
    else:
        scale = VIDEO_HEIGHT / img_height

    scale = min(scale, 2)

    new_width = int(img_width * scale)
    new_height = int(img_height * scale)

    img_resized = rgbimg.resize((new_width, new_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (new_width, new_height), (0, 0, 0))

    canvas.paste(img_resized, (0, 0))
    canvas.save(image_path, format=img.format)

def create_clip(panel):
    image_path = panel["image_path"]
    text = parse_text(panel["text"])
    
    preprocess_image(image_path)

    img = Image.open(image_path)
    img_width, img_height = img.size

    if img_width > VIDEO_WIDTH:
        move_range = img_width - VIDEO_WIDTH
    elif img_height > VIDEO_HEIGHT:
        move_range = img_height - VIDEO_HEIGHT
    else:
        move_range = 0

    duration = BASE_DURATION + move_range / MOVEMENT_SPEED
    image = ImageClip(image_path, duration=duration)

    if img_width > VIDEO_WIDTH:
        image = image.set_position(lambda t: (VIDEO_WIDTH-img_width + move_range * (t / duration), 'center'))
    elif img_height > VIDEO_HEIGHT:
        image = image.set_position(lambda t: ('center', VIDEO_HEIGHT-img_height + move_range * (t / duration)))
    else:
        image = image.set_position(('center', 'center'))

    text_chunks = chunkify_text(text)

    text_clips = []
    for idx, chunk in enumerate(text_chunks):
        if chunk.strip()=='':
            continue
        text_clip = TextClip(
            chunk,
            fontsize=60,
            color='White',
            bg_color='Transparent',
            font='Arial-Bold',
            size=(VIDEO_WIDTH, VIDEO_HEIGHT),
            stroke_color='Black', 
            stroke_width=5
        )
        text_clip = text_clip.set_position(('center', 'center')).set_duration(duration / len(text_chunks))
        text_clip = text_clip.set_start(idx * (duration / len(text_chunks)))
        
        text_clips.append(text_clip)

    final_clip = CompositeVideoClip([image] + text_clips, size=(VIDEO_WIDTH, VIDEO_HEIGHT))

    return final_clip

def page_short(page):
    clips = [create_clip(panel) for panel in page]
    final_video = concatenate_videoclips(clips, method="compose")
    return final_video

###############################################################
###                        UTILITIES                        ###
###############################################################

NEAR = 50

def remove_inconsitences(str):
    if str.isdigit():
        return ""
    inconsistences = ['NASSAN', 'SFX: ', 'SFX:']
    alternatives = [
        ['1D101', 'IDIOT']
    ]

    temp = str

    for i in inconsistences:
        temp = temp.replace(i, '')
    for i in alternatives:
        temp = temp.replace(i[0], i[1])

    return temp

def straighten_text(lines):
    bubbles = []
    current_bubble = []
    for line in lines:
        if len(current_bubble):
            gap = abs(current_bubble[-1]["MinTop"] - line["MinTop"])

            if gap > NEAR:
                bubbles.append(current_bubble)
                current_bubble = [line]
            else:
                current_bubble.append(line)
        else:
            current_bubble.append(line)
    
    bubbles.append(current_bubble)
    bubbles = list(map(lambda bubble: " ".join([line['LineText'] for line in bubble]), bubbles))
    
    return remove_inconsitences("\n".join(bubbles[::-1]))

def parse_text(obj):
    if isinstance(obj, str):
        return ""
     
    raw_text_data = obj["ParsedResults"][0]["TextOverlay"]["Lines"]
    text = straighten_text(raw_text_data)

    return text

def save_results_to_file(data, file_path):
    """
    Save pretty-printed data to a file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def chunkify_text(text):
    """
    Split the text into chunks. Each chunk will have 2-3 words, taking into account line breaks.
    """
    lines = text.split('\n')
    chunks = []

    for line in lines:
        words = line.split(' ')
        while len(words) > 0:
            chunk = ' '.join(words[:READ_SPEED])
            chunks.append(chunk)
            words = words[READ_SPEED:]  #

    return chunks

def display_detected_lines(img, lines):
    """
    Display the detected lines on the original image with colored lines.
    
    Parameters:
    - img: Original image where the lines will be drawn.
    - lines: Output from the get_long_lines function.
    """
    # Make sure the image is in color format (BGR)
    if len(img.shape) == 2:  # If the image is grayscale, convert it to BGR
        overlay_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        overlay_img = img.copy()
    
    # Draw lines for each detected line
    for line in lines:
        x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
        cv2.line(overlay_img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Green line, 2px thick
    
    # Resize for debugging purposes (optional)
    scale_percent = 50  # Resize to 50% of the original size
    width = int(overlay_img.shape[1] * scale_percent / 100)
    height = int(overlay_img.shape[0] * scale_percent / 100)
    resized_img = cv2.resize(overlay_img, (width, height), interpolation=cv2.INTER_AREA)
    
    # Display the image with the detected lines
    cv2.imshow("Detected Lines", resized_img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


def display_detected_panels(img, panels):
    """
    Display the contours of detected panels on the original image with red bounding boxes.
    
    Parameters:
    - img: Original image where the contours will be drawn.
    - panels: Output from the get_panels function.
    """
    # Make sure the image is in color format (BGR)
    if len(img.shape) == 2:  # If the image is grayscale, convert it to BGR
        overlay_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        overlay_img = img.copy()
    
    # Draw red rectangles for each panel
    for panel in panels:
        x, y, w, h = panel["x"], panel["y"], panel["w"], panel["h"]
        cv2.rectangle(overlay_img, (x, y), (x + w, y + h), (0, 0, 255, 2))  # Red rectangle, 2px thick
    
    # Resize for debugging purposes (optional)
    scale_percent = 50  # Resize to 50% of the original size
    width = int(overlay_img.shape[1] * scale_percent / 100)
    height = int(overlay_img.shape[0] * scale_percent / 100)
    resized_img = cv2.resize(overlay_img, (width, height), interpolation=cv2.INTER_AREA)
    
    # Display the image with red contours
    cv2.imshow("Detected Panels with Contours", resized_img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if PARSE:
        pages = [file for file in os.listdir(MANGA_PAGES_DIR) if file.endswith('.png')][2:3]
        results = []

        for i, page in enumerate(pages):
            data = process_manga_page(os.path.join(MANGA_PAGES_DIR, page), str(i))
            results.append(data)

        if GET_DATA:
            result_file = os.path.join("", "result.txt")
            save_results_to_file(results, result_file)
    
    else:
        with open("result.txt", "r", encoding="utf-8") as f:
            content = f.read()
            data = json.loads(content)
            clips = []

            for page in data[:4]:
                short = page_short(page)
                if short:
                    clips.append(short)
                    print(f'{len(clips)}/{len(data)}')

            final_video = concatenate_videoclips(clips, method="compose")
            print('Saving ...')
            final_video.write_videofile(
                os.path.join(RESULTS_DIR, MANGA_PAGES_DIR.split('/')[-1] + '.mp4'), 
                fps=24, 
                threads=8, 
                preset='fast'
            )

