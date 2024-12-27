import os
from moviepy.editor import *
from PIL import Image
import numpy as np

TARGET_ASPECT_RATIO = 9 / 16
VIDEO_WIDTH = 900
VIDEO_HEIGHT = int(VIDEO_WIDTH / TARGET_ASPECT_RATIO)
BASE_DURATION = 2

def gray_to_rgb(image_path):
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

    # Save the resized and padded image
    canvas.save(image_path, format=img.format)

def create_clip(image_path, base_duration):
    """
    Create a CompositeVideoClip that moves to show the entire image,
    with duration based on the movement range and base duration.
    """
    gray_to_rgb(image_path)  # Ensure the image is RGB and resized

    # Load the image to determine its dimensions
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Calculate movement range and duration
    if img_width > VIDEO_WIDTH:  # Image is wider than the frame
        move_range = img_width - VIDEO_WIDTH  # Horizontal movement range
    elif img_height > VIDEO_HEIGHT:  # Image is taller than the frame
        move_range = img_height - VIDEO_HEIGHT  # Vertical movement range
    else:
        move_range = 0  # No movement required

    # Duration is proportional to the movement range, with a base duration
    duration = base_duration + move_range / 400  # Adjust divisor for desired speed

    # Create the ImageClip
    image = ImageClip(image_path, duration=duration)

    # Set movement position
    if img_width > VIDEO_WIDTH:  # Horizontal movement
        image = image.set_position(lambda t: (VIDEO_WIDTH-img_width + move_range * (t / duration), 'center'))
    elif img_height > VIDEO_HEIGHT:  # Vertical movement
        image = image.set_position(lambda t: ('center', VIDEO_HEIGHT-img_height + move_range * (t / duration)))
    else:  # No movement needed
        image = image.set_position(('center', 'center'))

    # Return the CompositeVideoClip with adjusted position and duration
    return CompositeVideoClip([image], size=[VIDEO_WIDTH, VIDEO_HEIGHT])



def main(image_files, output_file="final_video.mp4"):
    # Process each image into a video clip
    clips = [create_clip(image_path, BASE_DURATION) for image_path in image_files]

    # Concatenate all clips into one final video
    final_video = concatenate_videoclips(clips, method="compose")
    final_video.write_videofile(output_file, fps=30)

# Input image files
image_files = [
    "./panels/panel_0.png", 
    "./panels/panel_1.png", 
    "./panels/panel_2.png", 
    "./panels/panel_3.png", 
    "./panels/panel_4.png", 
    "./panels/panel_5.png", 
    "./panels/panel_6.png", 
    "./panels/panel_7.png"
]

main(image_files)