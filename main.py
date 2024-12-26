import cv2
import numpy as np
import os
from PIL import Image
import pytesseract

def preprocess_image(image_path):
    """
    Preprocess the input black-and-white manga page to prepare for panel detection.
    """
    # Load the image (grayscale assumed for black-and-white image)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Invert colors: panels should be white on a black background for contour detection
    inverted = cv2.bitwise_not(img)
    # Apply binary thresholding
    _, thresh = cv2.threshold(inverted, 128, 255, cv2.THRESH_BINARY)
    return img, thresh

def detect_and_sort_panels(thresh, original_img):
    """
    Detect vignettes (panels) using contours and sort them in reading order.
    """
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panels = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out small noise by area
        if w * h > 5000:  # Adjust size threshold based on your manga page
            cropped_panel = original_img[y:y + h, x:x + w]
            panels.append({"x": x, "y": y, "w": w, "h": h, "panel_img": cropped_panel})
    
    # Sort panels: top-to-bottom, then left-to-right
    panels = sorted(panels, key=lambda p: (p["y"], p["x"]))
    return panels

def extract_text_from_panels(panels, output_dir="panels"):
    """
    Extract text from each detected panel using pytesseract and save each panel as a PNG file.
    """
    extracted_data = []
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, panel in enumerate(panels):
        # Convert panel image to PIL format for pytesseract
        panel_pil = Image.fromarray(panel["panel_img"])
        
        # Extract text using pytesseract
        text = pytesseract.image_to_string(panel_pil)
        
        # Save the panel as a PNG image
        panel_filename = os.path.join(output_dir, f"panel_{i}.png")
        cv2.imwrite(panel_filename, panel["panel_img"])
        
        # Append the extracted data along with the text and coordinates
        extracted_data.append({
            "index": i,
            "coordinates": (panel["x"], panel["y"], panel["w"], panel["h"]),
            "text": text.strip(),
            "image_path": panel_filename  # Path to the saved panel image
        })
    
    return extracted_data

def resize_image_to_fit_screen(image, max_width=1200, max_height=800):
    """
    Resize the image to fit within the screen dimensions.
    """
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    new_size = (int(width * scaling_factor), int(height * scaling_factor))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized

def process_manga_page(image_path):
    """
    Process the manga page to detect panels, sort them, and extract text from each.
    """
    # Preprocess the image
    original_img, thresh = preprocess_image(image_path)
    
    # Detect and sort panels
    panels = detect_and_sort_panels(thresh, original_img)
    
    # Extract text from panels
    extracted_data = extract_text_from_panels(panels)
    
    return original_img, panels, extracted_data

if __name__ == "__main__":
    # Path to your manga page image (black-and-white)
    manga_page_path = "C:/Users/Kei/Documents/GitHub/MangaReader/test.png"
    
    # Process the manga page
    original_img, panels, result = process_manga_page(manga_page_path)

    
    # Optional: Visualize detected panels (resized for screen)
    resized_img = resize_image_to_fit_screen(original_img)
    scaling_factor = resized_img.shape[1] / original_img.shape[1]

    resized_img_color = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)

    for panel in result:
        print(panel['text'])
        print("-" * 40)

    for panel in panels:
        x, y, w, h = panel["x"], panel["y"], panel["w"], panel["h"]
        # Scale coordinates to match the resized image
        x = int(x * scaling_factor)
        y = int(y * scaling_factor)
        w = int(w * scaling_factor)
        h = int(h * scaling_factor)
        # Draw red rectangle around each panel (color BGR: (0, 0, 255))
        cv2.rectangle(resized_img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Display the image with red rectangles
    cv2.imshow("Detected Panels", resized_img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()