import os
from moviepy.editor import *
import numpy as np
import requests
from autocorrect import Speller
import re
import cv2
from PIL import Image

MANGA_PAGES_DIR = r"C:\Users\Kei\Desktop\Rent-A-Girlfriend Chapter 358_files"
MANGA_PANELS_DIR = "panels"
API_KEY_FILE = "./key/api_key.txt"
OCR_API_URL = "https://api.ocr.space/parse/image"

###############################################################
###                     PANEL EXTRACTION                    ###
###############################################################

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

    spell = Speller()
    text = spell(text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.strip().replace('\r', ' ').replace('\n', ' ')

    return text

def get_text(img_path, api_key):
    with open(img_path, "rb") as f:
        response = requests.post(
            OCR_API_URL,
            files={"file": f},
            data={"apikey": api_key, "language": "eng"}
        )
    
    result = response.json()
    try:
        return result['ParsedResults'][0]['ParsedText']
    except:
        return ''

def retrieve_text(panels):
    api_key = load_api_key()
    texts = []

    for i in range(len(panels)):
        text = get_text(panels[i]["image_path"], api_key)
        cleaned_text = clean_text(text)
        texts.append(cleaned_text)

    return texts

def save_panels(panels, base_dir, num):
    """
    Extract text from each detected panel using pytesseract and save each panel as a PNG file.
    """
    extracted_data = []
    output_dir = base_dir+'/'+num
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, panel in enumerate(panels):
        # Save the panel as a PNG image
        panel_filename = os.path.join(output_dir, f"panel_{i}.png")
        cv2.imwrite(panel_filename, panel["panel_img"])

        # Append the extracted data along with the text and coordinates
        extracted_data.append({
            "index": i,
            "coordinates": (panel["x"], panel["y"], panel["w"], panel["h"]),
            "image_path": panel_filename  # Path to the saved panel image
        })
    
    return extracted_data

def process_manga_page(image_path, num):
    """
    Process the manga page to detect panels, sort them, and extract text from each.
    """

    original_img, thresh = preprocess_page(image_path)
    panels_contours = get_panels(thresh, original_img)
    panels = save_panels(panels_contours, MANGA_PANELS_DIR, num)
    texts = retrieve_text(panels)
    print(texts)
    
    return original_img, panels, texts



def load_api_key():
    """Load the API key from a secure file."""
    try:
        with open(API_KEY_FILE, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"API key file not found at {API_KEY_FILE}")
    except Exception as e:
        raise Exception(f"Error reading API key file: {e}")
if __name__ == "__main__":
    pages = [file for file in os.listdir(MANGA_PAGES_DIR) if file.endswith('.png')]
    for i in range(3):
        process_manga_page(MANGA_PAGES_DIR+'\\'+pages[i], str(i))

    
