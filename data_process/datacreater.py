import os
import csv
import json

import cv2
import glob

image_path = '/home/liu.11387/code/DATASET/jax_pair_facade/*_STREET_RGB.jpg'
conditional_image_path = '/home/liu.11387/code/DATASET/jax_pair_facade/*_SAT_RGB.png'
save_directory = './data_process/satellite_pair'

# Glob will return all the file paths that match the given pattern
image_files = glob.glob(image_path)
# Load all those images
images = [cv2.imread(file) for file in image_files]
image_paths = []
caption_list = []
for image, filepath in zip(images, image_files):
    filename = os.path.basename(filepath)  # Get only the filename from the full path
    image_paths.append(filename)
    caption_list.append('street-view, panorama image, high resolution')
    save_path = os.path.join(save_directory, filename)
    cv2.imwrite(save_path, image)

conditional_image_paths = []
conditional_image_files = [image_path.replace('_STREET_RGB.jpg', '_SAT_RGB.png') for image_path in image_files]
# Load all those images
conditional_images = [cv2.imread(file) for file in conditional_image_files]
for image, filepath in zip(conditional_images, conditional_image_files):
    filename = os.path.basename(filepath)  # Get only the filename from the full path
    conditional_image_paths.append(filename)
    save_path = os.path.join(save_directory, filename)
    cv2.imwrite(save_path, image)

with open('./data_process/satellite_pair/metadata.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image", "text", "conditioning_image"])

    for l1, l2, l3 in zip(image_paths, caption_list, conditional_image_paths):
        writer.writerow([l1, l2, l3])

with open('./data_process/satellite_pair/metadata.json', 'w') as json_file:
    meta_data = []
    for l1, l2, l3 in zip(image_paths, caption_list, conditional_image_paths):
        data = {
            "image": l1,
            "text": l2,
            "conditioning_image": l3,
        }
        meta_data.append(data)

    json.dump(meta_data, json_file, indent=4)