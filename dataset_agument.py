import os
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
from shutil import copyfile
from skimage.util import random_noise
import random


def load_copy_file():
    # load dataset
    original_images = r'G:\大学\大四\Project\archive\images'
    original_csv = r'G:\大学\大四\Project\archive\annotations.csv'
    new_images = r'G:\大学\大四\毕业论文\数据集\数据扩充数据\images'
    new_csv = r'G:\大学\大四\毕业论文\数据集\数据扩充数据\augmented_annotations.csv'

    # copy original file to new file
    if not os.path.exists(new_images):
        os.makedirs(new_images)

    copyfile(original_csv, new_csv)
    for filename in os.listdir(original_images):
        src_path = os.path.join(original_images, filename)
        dst_path = os.path.join(new_images, filename)
        copyfile(src_path, dst_path)

    return new_images, new_csv


def augment_image(image, new_images_folder, file_name_prefix, category, augmented_data):
    # add random noisy to image
    noisy_image = random_noise(np.array(image), mode='gaussian', var=0.01 ** 2)
    noisy_image = (255 * noisy_image).astype(np.uint8)
    noisy_image = Image.fromarray(noisy_image)

    # rotated image in random angle
    angle = random.randint(-10, 10)
    rotated_image = noisy_image.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))

    #  random scale image
    scale_factor = random.uniform(0.8, 1.2)
    new_size = (int(scale_factor * rotated_image.size[0]), int(scale_factor * rotated_image.size[1]))
    # new_size = tuple([int(scale_factor * s) for s in rotated_image.size])
    scaled_image = rotated_image.resize(new_size, resample=Image.BICUBIC)

    # change brightness
    brightness_factor = random.uniform(0.8, 1.2)
    brightness_image = ImageEnhance.Brightness(scaled_image).enhance(brightness_factor)

    # save
    new_file_name = file_name_prefix + '_create.png'
    brightness_image.save(os.path.join(new_images_folder, new_file_name))
    augmented_data.append([new_file_name, category])

    return augmented_data


# Create and open csv file
new_images_path, new_csv_path = load_copy_file()
df = pd.read_csv(new_csv_path)

# argument image to 150
while True:
    # select category less than 150
    selected_df = df[df.groupby('category')['category'].transform('count') < 150]

    # if all more than, then end
    if selected_df.empty:
        break

    Augmented_data = []
    for index, row in selected_df.iterrows():
        image_path = os.path.join(new_images_path, row['file_name'])
        image_obj = Image.open(image_path)
        if image_obj is None:
            continue

        file_name = os.path.splitext(row['file_name'])[0]
        Augmented_data = augment_image(image_obj, new_images_path, file_name, row['category'], Augmented_data)

    # save to new CSV file
    augmented_df = pd.DataFrame(Augmented_data, columns=['file_name', 'category'])
    df = pd.concat([df, augmented_df])

# save to CSV file
df.to_csv(new_csv_path, index=False)
