import os
import requests
from PIL import Image
import numpy as np
from autocorrect import Speller
import re

# Configuration for OCR.Space API
API_KEY_FILE = "./key/api_key.txt"  # Path to the file containing your API key
OCR_API_URL = "https://api.ocr.space/parse/image"

# Function to load the API key securely
def load_api_key():
    """Load the API key from a secure file."""
    try:
        with open(API_KEY_FILE, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"API key file not found at {API_KEY_FILE}")
    except Exception as e:
        raise Exception(f"Error reading API key file: {e}")

from PIL import Image
import os

def preprocess_image(img_path):
    """Preprocess the image for better OCR detection and ensure size is within limits."""
    img = Image.open(img_path)
    
    # Convert to grayscale
    img = img.convert("L")

    # Resize the image to a smaller target width while keeping the aspect ratio
    target_width = 1200
    img_width, img_height = img.size
    if img_width > target_width:
        aspect_ratio = img_height / img_width
        target_height = int(target_width * aspect_ratio)
        img = img.resize((target_width, target_height))

    # Ensure the final image size is within 1024 KB
    temp_path = img_path.replace(".png", "_compressed_temp.png")
    quality = 95  # Start with high quality
    while True:
        img.save(temp_path, format="PNG", quality=quality)
        file_size_kb = os.path.getsize(temp_path) / 1024  # Get file size in KB
        if file_size_kb <= 1024 or quality <= 10:  # Minimum quality threshold
            break
        quality -= 5  # Reduce quality to lower the file size

    # Reload the compressed image to ensure compatibility with further steps
    img = Image.open(temp_path)
    os.remove(temp_path)  # Clean up the temporary compressed file
    return img


# Function to clean and formalize the detected text
def clean_text(text):
    """Clean and formalize the text."""
    # Correct common spelling issues using autocorrect
    spell = Speller()
    text = spell(text)

    # Manually replace common mismatches (custom rules)
    text = text.replace("8dmp", "BDMP").replace("VOUR", "YOUR").replace("sO", "SO")
    
    # Use regular expressions to remove unwanted characters or fix patterns
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = text.strip()  # Clean leading and trailing spaces
    return text

# Function to detect text and save it to a file
def detect_and_save_text(panel_path, output_path, api_key):
    """
    Detect text from an image and save the output to a text file.
    Uses OCR.Space API for detection.
    """
    try:
        # Preprocess the image
        img = preprocess_image(panel_path)
        # Save the preprocessed image temporarily
        temp_path = panel_path.replace(".png", "_temp.png")
        img.save(temp_path)

        # Send the image to the OCR.Space API
        with open(temp_path, "rb") as f:
            response = requests.post(
                OCR_API_URL,
                files={"file": f},
                data={"apikey": api_key, "language": "eng"}
            )
        
        # Parse the response
        result = response.json()
        if response.status_code == 200 and "ParsedResults" in result:
            detected_text = result["ParsedResults"][0]["ParsedText"].strip()

            # Clean and formalize the detected text
            cleaned_text = clean_text(detected_text)

            # Save the cleaned text to a file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            print(f"Text extracted and saved: {output_path}")
        else:
            print(f"Failed to extract text for {panel_path}: {result.get('ErrorMessage', 'Unknown error')}")
        
        # Remove the temporary file
        os.remove(temp_path)
    except Exception as e:
        print(f"Error processing {panel_path}: {e}")

# Main function to process all images in the /panels directory
def main():
    input_dir = "./panels"
    output_dir = "./text"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the API key
    api_key = load_api_key()

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            panel_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
            detect_and_save_text(panel_path, output_path, api_key)

if __name__ == "__main__":
    main()
