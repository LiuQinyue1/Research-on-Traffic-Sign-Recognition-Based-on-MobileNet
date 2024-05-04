import os
import numpy as np
import pandas as pd

from PIL import Image, ImageFilter
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


# Load dataset and csv file
def load_dataset(folder_path, csv_path):
    class_name = get_actual_classes()
    annotations_df = pd.read_csv(csv_path)
    image_paths = [os.path.join(folder_path, filename) for filename in annotations_df['file_name']]
    labels = annotations_df['category'].values
    labels_categorical = to_categorical(labels)
    return image_paths, labels, labels_categorical, class_name


# Process method 1
def processing_1_paste(image_paths, target_size=(224, 224), padding_color=(255, 255, 255)):
    for image_path in image_paths:
        try:
            image = Image.open(image_path)
            original_size = image.size

            padded_image = Image.new("RGB", target_size, padding_color)

            # Calculate the picture location
            left = (target_size[0] - original_size[0]) // 2
            top = (target_size[1] - original_size[1]) // 2

            # Paste original into new picture
            padded_image.paste(image, (left, top))
            denoised_image = padded_image.filter(ImageFilter.MedianFilter(size=3))

            # Preprocess picture
            img_array = preprocess_input(np.array(denoised_image))

            yield img_array
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")


# Process method 2--4
def processing_2_interpolation(image_paths, target_size=(224, 224), padding_color=(255, 255, 255)):

    for image_path in image_paths:
        try:
            image = Image.open(image_path)
            original_size = image.size

            # calculate picture size
            ratio = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))

            image = image.resize(new_size, Image.LANCZOS)  # Bilinear, Bicubic, Lanczos
            padded_image = Image.new("RGB", target_size, padding_color)

            # calculate the picture location and paste
            left = (target_size[0] - new_size[0]) // 2
            top = (target_size[1] - new_size[1]) // 2
            padded_image.paste(image, (left, top))

            # Apply denoising using median filter
            denoised_image = padded_image.filter(ImageFilter.MedianFilter(size=3))

            # preprocess picture
            img_array = preprocess_input(np.array(denoised_image))

            yield img_array
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")


# Data Augmentation
def data_generate(x_train, y_train, x_val, y_val, batch_size):
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest',
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=None
    )

    # Generate augmented training data and valid data
    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size)

    return train_generator, val_generator


# Get actual classes for different number
def get_actual_classes():
    class_name = [
        '1: Speed Limit 5',
        '2: Speed Limit 15',
        '3: Speed Limit 30',
        '4: Speed Limit 40',
        '5: Speed Limit 50',
        '6: Speed Limit 60',
        '7: Speed Limit 70',
        '8: Speed Limit 80',
        '9: No Straight and Left Turn',
        '10: No Straight and Right Turn',
        '11: No Straight Ahead',
        '12: No Left Turn',
        '13: No Left or Right Turn',
        '14: No Right Turn',
        '15: No Overtaking',
        '16: No U-turn',
        '17: No Entry',
        '18: No Horn',
        '19: End of 40 Speed Limit',
        '20: End of 50 Speed Limit',
        '21: Straight and Left Turn Allowed',
        '22: Straight Ahead Allowed',
        '23: Left Turn Allowed',
        '24: Left and Right Turn Allowed',
        '25: Right Turn Allowed',
        '26: Left Lane',
        '27: Right Lane',
        '28: Roundabout',
        '29: Small Car Lane',
        '30: Horn Allowed',
        '31: Bicycle Lane',
        '32: U-turn Ahead',
        '33: Beware of Median Island',
        '34: Traffic Lights Ahead',
        '35: Caution',
        '36: Pedestrian Crossing Ahead',
        '37: Beware of Bicycles',
        '38: School Ahead',
        '39: Right Curve Ahead',
        '40: Left Curve Ahead',
        '41: Downhill Ahead',
        '42: Uphill Ahead',
        '43: Slow Down',
        '44: Right Intersection',
        '45: Left Intersection',
        '46: Residential Area',
        '47: Winding Road',
        '48: Railway Crossing',
        '49: Construction Ahead',
        '50: Series of Turns',
        '51: Beware of Barriers',
        '52: Risk of Tailgating',
        '53: Stop',
        '54: No Vehicles Allowed',
        '55: No Parking',
        '56: No Entry',
        '57: Yield to Pedestrians',
        '58: Stop for Inspection'
    ]

    return class_name
