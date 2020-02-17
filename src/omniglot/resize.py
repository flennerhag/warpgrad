import argparse
import glob
from PIL import Image
from tqdm import tqdm


def run(path, height, width):
    """Resize images"""
    print('resizing images')
    all_images = glob.glob(path + '*')
    for image_file in tqdm(all_images):
        im = Image.open(image_file)
        im = im.resize((height, width), resample=Image.LANCZOS)
        im.save(image_file)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to resize a set of images in a folder')
    parser.add_argument(
        "-f", "--folder", type=str,
        help="Path to folder containing the images to resize (it can contain "
             "the special character '*' to refer to a set of folders).")
    parser.add_argument(
        "-H", "--height", type=int, help="Height of the resized image")
    parser.add_argument(
        "-W", "--width", type=int, help="Width of the resized image")
    args = parser.parse_args()
    run(args.folder, args.height, args.width)
