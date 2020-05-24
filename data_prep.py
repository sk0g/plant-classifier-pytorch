#!/usr/bin/env python3
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from random import sample, shuffle

from PIL import Image, ImageStat


def calculate_number_of_validation_images(n):
    """
    Calculate number of validation images that should be selected, given an image pool of size n
    """

    return n // 6  # 15% of dataset for validation, as the previous method produced too few validation images


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


def resize_file(tif_filepath):
    """
    Does the actual resizing and conversion of the images, provided a relative filepath to the TIF file

    Conversions done are: CMYK -> RGB (if needed), TIF -> PNG
    Shrinks them down to height/3, width/3 (1/9th the pixels)
    """
    png_filepath = tif_filepath.replace('.tif', '.png')
    if os.path.isfile(png_filepath):
        return

    im = Image.open(tif_filepath)

    (x, y) = im.size

    new_x = round(x / 3)
    new_y = round(y / 3)

    if im.mode == "CMYK":
        im = im.convert("RGB")

    im.thumbnail((new_x, new_y), Image.LANCZOS)
    im.save(png_filepath)


def resize_and_convert_images_to_png():
    """
    For each file in the folders:
        Convert to a .PNG file, preserve file size
        Label it something computer-processable

    """
    print("Running resize_and_convert_images_to_png()")

    current_directory = './Deep Learning Plant Classifier'

    # first pass - convert to png, preserving file name (except for the extension)
    for (root, _, files) in os.walk(current_directory):
        tif_filepaths = [rf"{root}\{f}" for f in files if f.endswith(".tif")]

        with ThreadPoolExecutor(max_workers=16) as executor:
            executor.map(resize_file, tif_filepaths)

    # second pass - delete the tif file IF PNG EXISTS
    for (root, _, files) in os.walk(current_directory):
        for file_name in [f for f in files if f.endswith(".tif")]:
            file_path = rf"{root}\{file_name}"
            png_filepath = file_path.replace('.tif', '.png')

            if os.path.isfile(png_filepath):
                os.remove(file_path)

    print("All done! PNG images are ready to be split into fragments now.")


def generate_bounds_for_fragments(x_size, y_size, move_size, image_dimension):
    """
    Generate bounds for fragments, for an image of arbitrary size

    Inputs:
        x_size - width of the image
        y_size - height of the image
        move_size - pixels to move (horizontally and vertically) between each step
    Returns:
        a list of 4-tuples, of the format (x_start, y_start, x_end, y_end)
    """
    bounds = []

    moves_x = (x_size - image_dimension) // move_size
    moves_y = (y_size - image_dimension) // move_size

    for y in range(moves_y):
        for x in range(moves_x):
            y_start = y * move_size
            x_start = x * move_size
            x_end = x_start + image_dimension
            y_end = y_start + image_dimension
            bounds.append((x_start, y_start, x_end, y_end))

    return bounds


def split_images_into_fragments():
    """
    Split each PNG image into multiple 400*400 images
        The exact amount should go off the size of the image
        The size will vary as, after resizing and converting images, they are also cropped to only include plant bits

    NOTE: images should be manually re-checked after splitting, as some may contain largely un-necessary info,
    which would hamper training
    """
    print("Running split_images_into_fragments()")

    current_directory = './Deep Learning Plant Classifier'

    for (root, _, files) in os.walk(current_directory):
        for file_name in [f for f in files if f.endswith(".png")]:
            file_path = rf"{root}\{file_name}"

            img = Image.open(file_path)
            (x, y) = img.size

            image_bounds = generate_bounds_for_fragments(x, y, 200, 400)

            for fragment_number, bounds in enumerate(image_bounds):
                fragment = img.crop(bounds)
                fragment_name = file_path.replace(
                    ".png", f",fragment-{fragment_number:04}.png")
                fragment.save(fragment_name)

    print("All done! PNG fragments have been generated. Once junk images are deleted, you can create variants.")


def generate_fragment_variants():
    """
    Generates variants for each image fragment, and also resizes them to the target size (224 * 224)

    Each fragment should also have variants, as below:
        A - rotated 90 degrees
        B - rotated 180 degrees
        C - rotated 270 degrees
        D - flipped horizontally
        E - flipped vertically
        Z - resize only
    """
    print("Running generate_fragment_variants()")

    current_directory = '../dataset'
    new_dimensions = (224, 224)

    for (root, _, files) in os.walk(current_directory):
        for file_name in [f for f in files if f.endswith(".png") and "fragment" in f and "variant" not in f]:
            file_path = rf"{root}\{file_name}"

            img = Image.open(file_path)

            """ 
            NOTE: These transforms can be done during batch loading with torchvision.transforms, 
            so this function is only used to resize to (224, 244)
            """
            # # Variant A
            # a = img.transpose(Image.ROTATE_90)
            # a.thumbnail(new_dimensions, Image.LANCZOS)
            # a.save(file_path.replace('.png', ',variant-a.png'))

            # # Variant B
            # b = img.transpose(Image.ROTATE_180)
            # b.thumbnail(new_dimensions, Image.LANCZOS)
            # b.save(file_path.replace('.png', ',variant-b.png'))

            # # Variant C
            # c = img.transpose(Image.ROTATE_270)
            # c.thumbnail(new_dimensions, Image.LANCZOS)
            # c.save(file_path.replace('.png', ',variant-c.png'))

            # # Variant D
            # d = img.transpose(Image.FLIP_LEFT_RIGHT)
            # d.thumbnail(new_dimensions, Image.LANCZOS)
            # d.save(file_path.replace('.png', ',variant-d.png'))

            # # Variant E
            # e = img.transpose(Image.FLIP_TOP_BOTTOM)
            # e.thumbnail(new_dimensions, Image.LANCZOS)
            # e.save(file_path.replace('.png', ',variant-e.png'))

            # Variant Z - resize only
            z = img
            z.thumbnail(new_dimensions, Image.LANCZOS)
            z.save(file_path.replace('.png', 'variant-z.png'))

    print("All done! Almost training time :)")


def generate_and_record_splits():
    """
    Generates training, validation and testing splits for images in ../dataset/class_name/[image_names]

    Stores these variants under filenames in the scheme of plan-0.json, plan-1.json... plan-9.json

    The basic format of the JSON file will be as follows:
    {
        "background": {
            "trainingVariants": [
                "file1,fragment-0001,variant-b.png",
                "file2,fragment-0001,variant-z.png",
                "file1-fragment-0001,variant-d.png",
            ],
            "validationVariants": [
                "file1,fragment-0001,variant-a.png",
                "file2,fragment-0001,variant-b.png",
                "file1-fragment-0001,variant-c.png",
            ]
            "testingFiles": [
                "file3"
            ]
        },
        "bertya" {
            "trainingVariants": [
                "file4,fragment-0001,variant-b.png",
                "file5,fragment-0001,variant-z.png",
                "file4-fragment-0001,variant-d.png",
            ],
            "validationVariants": [
                "file4,fragment-0001,variant-a.png",
                "file5,fragment-0001,variant-b.png",
                "file4-fragment-0001,variant-c.png",
            ]
            "testingFiles": [
                "file6"
            ]
        }
        ...
    }

    NOTE: Testing allocates entire files at a time, while training/ validation splits allocate on the variant level.
    Any fragment and variant generated from a file name under testingFiles will be used for testing,
    while training and validation fragments and variants will be accessed exactly as per the split.
    """

    dataset_directory = '../dataset'
    batch_directory = '../batches'

    try:
        shutil.rmtree(batch_directory)
        os.rmdir(batch_directory)
    except:
        print("batches directory not found, continuing")

    os.mkdir(batch_directory)
    print("Created batches directory")

    for split_number in range(0, 10):
        split_filename = f"split-{split_number}.json"
        print(f"Generating {split_filename}")

        """
        Create split directory with files in the following structure:
        ../dataset/
        ../batches/
            | - batch-0/
            | - train/
            | - val/
            | - test/
            | - batch-1/
            | - ...
        """
        split_directory = f"{batch_directory}/batch-{split_number}"
        train_path = f"{split_directory}/train"
        test_path = f"{split_directory}/test"
        val_path = f"{split_directory}/val"

        for directory_name in [split_directory, train_path, test_path, val_path]:
            os.mkdir(directory_name)

        split_details = dict()

        for (root, _, files) in os.walk(dataset_directory):
            # Only iterate through folders containing variant images
            if any("variant-z.png" in s for s in files):
                # Top level JSON key under which to write data
                class_name = root.split(os.sep)[-1]

                # Make a set from just the file names (image names are of the format image_name,X,Y.png)
                unique_image_names = {file.split(",")[0]
                                      for file in files if file.endswith("variant-z.png")}

                # Dedicate 10% of images to testing, or 1, whichever is higher
                test_image_count = max(1, round(len(unique_image_names) / 10))
                test_image_names = sample(unique_image_names, test_image_count)

                # Use variants from images not in test_image_names for training and validation
                usable_variants = [f for f in files if "variant-z" in f and
                                   f.split(",")[0] not in test_image_names]

                validation_variant_count = calculate_number_of_validation_images(
                    len(usable_variants))

                validation_variants = []
                # Pop the required number of variants for the validation set
                for _ in range(0, validation_variant_count):
                    shuffle(usable_variants)
                    validation_variants.append(usable_variants.pop())
                # The remaining variants in usable_variants are the training variants

                split_details[class_name] = {
                    "testingFiles": test_image_names,
                    "trainingVariants": usable_variants,
                    "validationVariants": validation_variants
                }

                # Write split details into JSON file
                with open(split_filename, 'w+') as f:
                    json.dump(split_details, f, indent=4)

                # Create class directory for current class
                for directory in [train_path, test_path, val_path]:
                    os.mkdir(f"{directory}/{class_name}")

                # Copy files to directories
                for train_variant in usable_variants:
                    shutil.copyfile(src=f"{dataset_directory}/{class_name}/{train_variant}",
                                    dst=f"{train_path}/{class_name}/{train_variant}")
                print(f"Copied images to batch-{split_number}/train")

                for val_variant in validation_variants:
                    shutil.copyfile(src=f"{dataset_directory}/{class_name}/{val_variant}",
                                    dst=f"{val_path}/{class_name}/{val_variant}")
                print(f"Copied images to batch-{split_number}/val")

                for image_name in test_image_names:
                    variants_to_copy = [f for f in files if image_name in f]

                    for test_variant in variants_to_copy:
                        shutil.copyfile(src=f"{dataset_directory}/{class_name}/{test_variant}",
                                        dst=f"{test_path}/{class_name}/{test_variant}")
                print(f"Copied images to batch-{split_number}/test")

    print("All done! You can now load up the splits and begin training.")


def calculate_weights():
    dataset_directory = '../dataset'

    weights = []
    for (_, _, files) in os.walk(dataset_directory):
        image_count = len([f for f in files if f.endswith("-z.png")])
        if image_count > 0:
            weights.append(1000 / max(1, len(files)))

    print(f"Weight calculation complete. Found all classes: {len(weights) == 17}")
    print(weights)


def calculate_mean_and_std():
    dataset_directory = '../dataset'

    mean_list = [[], [], []]
    std_list = [[], [], []]
    for (root, _, files) in os.walk(dataset_directory):
        images = [f for f in files if f.endswith("-z.png")]
        if len(images) > 0:
            for image_name in images:
                stats = ImageStat.Stat(
                    Image.open(
                        os.path.join(root, image_name)))

                mean = stats.mean
                std = stats.stddev

                for i in range(3):
                    mean_list[i].append(mean[i])
                    std_list[i].append(std[i])

    calculated_mean = (
        (sum(mean_list[0]) / len(mean_list[0])) / 255,
        (sum(mean_list[1]) / len(mean_list[1])) / 255,
        (sum(mean_list[2]) / len(mean_list[2])) / 255,
    )
    calculated_std = (
        (sum(std_list[0]) / len(std_list[0])) / 255,
        (sum(std_list[1]) / len(std_list[1])) / 255,
        (sum(std_list[2]) / len(std_list[2])) / 255,
    )

    print("Mean and standard deviation calculated")
    print(f"Mean: {calculated_mean} | Std: {calculated_std}")


if __name__ == '__main__':
    prompt_text = "What function should be run?  \n" \
                  "[c]heck files | " \
                  "[r]esize and convert | " \
                  "[s]plit into fragments | " \
                  "[g]enerate fragment variants | " \
                  "s[p]lit dataset | " \
                  "calculate [w]eights | " \
                  "calculate [m]ean and std\n"

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
        elif key == "p":
            generate_and_record_splits()
        elif key == "w":
            calculate_weights()
        elif key == "m":
            calculate_mean_and_std()
        else:
            print("Unkown key pressed, try again?")
