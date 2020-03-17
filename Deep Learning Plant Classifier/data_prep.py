#!/usr/bin/env python3
import os


def check_files():
    """
    Check files so that:
        Each folder must have at least 2 (TIF) files
            Each file should not be empty
    If a folder is empty or only has one file, error log it

    Assumes the images are all under ./images/$plant_type/*.tif (file name is mostly unimportant)
    """
    print("Running check_files()")
    current_directory = './Deep Learning Plant Classifier'

    subdirectories = {}
    for (root, _, files) in os.walk(current_directory):
        if len([f for f in files if f.endswith(".tif")]) > 0:
            subdirectories[root] = len(files)

    folder_without_enough_files_exists = False
    for (k, v) in subdirectories.items():
        if v <= 1:
            print(f"{k} does not have enough TIF files under it - {v}")
            folder_without_enough_files_exists = True

    if folder_without_enough_files_exists:
        print("Please download more test data for the folder(s) that threw an error message above, and then proceed")
    else:
        print("All good! You can proceed to the next step now :)")


def resize_and_convert_images_to_png():
    """
    For each file in the folders:
        Convert to a uniform size, let's say, 6MP (3000*2000, if that exact aspect ratio is matched)
        Label it something computer-processable

    """
    print("Running resize_and_convert_images_to_png()")

    # first pass - convert to png, preserving file name (except for the extension)

    # second pass - delete the tif file IF PNG EXISTS


def split_images_into_fragments():
    """
    Split each PNG image into multiple 400*400 images
        The exact amount should go off the size of the image 
        The size will vary as, after resizing and converting images, they are also cropped to only include plant bits

    NOTE: images should be manually re-checked after splitting, as some may contain largely un-necessary info, 
    which would hamper training
    """
    print("Running split_images_into_fragments()")


def generate_fragment_variants():
    """
    Generates variants for each image fragment

    TODO: describe further
    """
    print("Running generate_fragment_variants()")


if __name__ == '__main__':
    prompt_text = "What function should be run? \n [c]heck files | [r]esize and convert | [s]plit into fragments | [g]enerate fragment variants\n"

    while True:
        key = input(prompt_text).lower()

        if key == "c":
            check_files()
        elif key == "r":
            resize_and_convert_images_to_png()
        elif key == "s":
            split_images_into_fragments()
        elif key == "g":
            generate_fragment_variants()
        else:
            print("Unkown key pressed, try again?")
