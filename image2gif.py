# https://www.blog.pythonlibrary.org/2021/06/23/creating-an-animated-gif-with-python/
import glob
from PIL import Image
from PIL import GifImagePlugin
GifImagePlugin.LOADING_STRATEGY = GifImagePlugin.LoadingStrategy.RGB_AFTER_DIFFERENT_PALETTE_ONLY

def make_gif(frame_folder):
    image_list = [image for image in glob.glob(f"{frame_folder}/*.jpg")]
    image_list.sort()
    frames = [Image.open(image) for image in image_list]

    frame_one = frames[0]
    
    import os
    if not os.path.exists("./gifs"):
        os.makedirs("./gifs")

    frame_one.save("./gifs/output.gif", 
                   format="GIF", 
                   append_images=frames,
                   save_all=True, 
                   duration=0.04, 
                   loop=0)
    
if __name__ == "__main__":
    make_gif("./rendered_images/")