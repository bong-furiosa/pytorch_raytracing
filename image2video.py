# pip3 install imageio
# pip3 install imageio[ffmpeg]
import imageio
import os

# Folder containing images
image_folder = './rendered_images/'
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort()

import os
if not os.path.exists("./videos"):
    os.makedirs("./videos")

# Create a video writer object
writer = imageio.get_writer('./videos/output.mp4', fps=24)

for image in images:
    img = imageio.imread(os.path.join(image_folder, image))
    writer.append_data(img)

writer.close()
