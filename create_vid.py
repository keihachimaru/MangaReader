import os
from moviepy.editor import *
from PIL import Image
import numpy as np

TARGET_ASPECT_RATIO = 9 / 16
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = int(VIDEO_WIDTH / TARGET_ASPECT_RATIO)
BASE_DURATION = 1

def gray_to_rgb(image_path):
    """Convert the image to RGB and resize it to fit one of the dimensions (width or height) of the target aspect ratio."""
    formatter = {"PNG": "RGBA", "JPEG": "RGB"}
    img = Image.open(image_path)
    
    rgbimg = Image.new(formatter.get(img.format, 'RGB'), img.size)
    rgbimg.paste(img)

    img_width, img_height = img.size

    if (img_width / img_height) > TARGET_ASPECT_RATIO:
        scale = VIDEO_WIDTH / img_width
    else:
        scale = VIDEO_HEIGHT / img_height

    new_width = int(img_width * scale)
    new_height = int(img_height * scale)

    img_resized = rgbimg.resize((new_width, new_height), Image.Resampling.LANCZOS)
    print(new_width, new_height)
    canvas = Image.new("RGB", (new_width, new_height), (0, 0, 0))

    canvas.paste(img_resized, (0, 0))

    # Save the resized and padded image
    canvas.save(image_path, format=img.format)
image_files = [
    "./panel_1/panel_0.png", 
    "./panel_1/panel_1.png", 
    "./panel_1/panel_2.png", 
    "./panel_1/panel_3.png", 
    "./panel_1/panel_4.png", 
    "./panel_1/panel_5.png", 
    "./panel_1/panel_6.png", 
    "./panel_1/panel_7.png"
]

i = 2
image_path = image_files[i]
gray_to_rgb(image_path)
image = ImageClip(image_path, duration=5)
image = image.set_position(lambda t: (t * (VIDEO_WIDTH/5),'center'))
image.fps = 30
composite = CompositeVideoClip([image], size=[VIDEO_WIDTH, VIDEO_HEIGHT])
composite.write_videofile('video.mp4')