#!/usr/bin/env python3


def check_files():
    """
    Check files so that:
        Each folder must have at least 2 (TIF) files
            Each file should not be empty
    If a folder is empty or only has one file, error log it

    Assumes the images are all under ./images/$plant_type/*.tif (file name is mostly unimportant)
    """
    pass


def resize_and_convert_images_to_png():
    """
    For each file in the folders:
        Convert to a uniform size, let's say, 6MP (3000*2000, if that exact aspect ratio is matched)
        Label it something computer-processable

    """
    pass


def split_png_images_for_training():
    """
    Split each PNG image into multiple 400*400 images
        The exact amount should go off the size of the image 
        The size will vary as, after resizing and converting images, they are also cropped to only include plant bits

    NOTE: images should be manually re-checked after splitting, as some may contain largely un-necessary info, 
    which would hamper training
    """
    pass


if __name__ == '__main__':
    prompt_text = "What function should be run? \n [c]heck files | [r]esize and convert | [s]plit for training\n"

    while True:
        key = input(prompt_text).lower()

        if key == 'c':
            check_files
        elif key == 'r':
            resize_and_convert_images_to_png()
        elif key == 's':
            split_png_images_for_training()
        else:
            print("Unkown key pressed, try again?")
