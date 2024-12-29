import os
import pprint
from moviepy.editor import *
import numpy as np
import requests
from autocorrect import Speller
import re
import cv2
from PIL import Image
import json

###############################################################
###                     PANEL EXTRACTION                    ###
###############################################################

MANGA_PAGES_DIR = "C:/Users/Kei/Desktop/Kokou no Hito Chapter 1"
MANGA_PANELS_DIR = "panels"
VIDEO_PAGES_DIR = "pages"
API_KEY_FILE = "./key/api_key2.txt"
OCR_API_URL = "https://api.ocr.space/parse/image"

def preprocess_page(path):
    """
    Preprocess the input black-and-white manga page to prepare for panel detection.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    inverted = cv2.bitwise_not(img)
    _, thresh = cv2.threshold(inverted, 128, 255, cv2.THRESH_BINARY)
    return img, thresh

def get_panels(thresh, img):
    """
    Detect vignettes (panels) using contours and sort them in reading order.
    If more than 8 panels are detected, return the whole image as one panel.
    """
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panels = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 5000:
            cropped_panel = img[y:y + h, x:x + w]
            panels.append({"x": x, "y": y, "w": w, "h": h, "panel_img": cropped_panel})

    if len(panels) > 8:
         panels = [{"x": 0, "y": 0, "w": img.shape[1], "h": img.shape[0], "panel_img": img}]

    panels = sorted(panels, key=lambda p: (p["y"], -p["x"]))
    
    return panels

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
    panels_contours = get_panels(thresh, original_img)

    panels = save_panels(panels_contours, num)
    retrieve_text(panels)

    return panels

###############################################################
###                      VIDEO CREATION                     ###
###############################################################

TARGET_ASPECT_RATIO = 9 / 16
VIDEO_WIDTH = 900
VIDEO_HEIGHT = int(VIDEO_WIDTH / TARGET_ASPECT_RATIO)
BASE_DURATION = 2
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
        text_clip = TextClip(
            chunk,
            fontsize=75,
            color='Red',
            bg_color='Transparent',
            font='Arial-Bold',
            size=(VIDEO_WIDTH, VIDEO_HEIGHT),
            stroke_color='White', 
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
        words = line.split()
        while len(words) > 0:
            chunk = ' '.join(words[:READ_SPEED])
            chunks.append(chunk)
            words = words[READ_SPEED:]  #

    return chunks

if __name__ == "__main__":
    parse = False
    if parse:
        pages = [file for file in os.listdir(MANGA_PAGES_DIR) if file.endswith('.png')][:11]
        results = []

        for i, page in enumerate(pages):
            data = process_manga_page(os.path.join(MANGA_PAGES_DIR, page), str(i))
            results.append(data)

        result_file = os.path.join("", "result.txt")
        save_results_to_file(results, result_file)
    
    else:
        with open("result.txt", "r", encoding="utf-8") as f:
            content = f.read()
            data = json.loads(content)
            clips = []

            for page in data:
                short = page_short(page)
                if short:
                    clips.append(short)
                    print(f'{len(clips)}/{len(data)}')

            final_video = concatenate_videoclips(clips, method="compose")
            final_video.write_videofile(
                MANGA_PAGES_DIR.split('/')[-1] + '.mp4', 
                fps=30, 
                threads=4, 
                preset='ultrafast'
            )

    
