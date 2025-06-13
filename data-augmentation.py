import os
import tensorflow as tf
from tensorflow.keras import layers

# This script loads and preprocesses images for data augmentation.
def load_and_preprocess_image(image_path):
    # Load the image from the specified path
    image = tf.io.read_file(image_path)
    
    # Decode the image as a JPEG file, ensuring it has 3 channels (RGB)
    image = tf.image.decode_jpeg(image, channels=3)
    
    # Convert between data types, scaling the values appropriately before casting
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

# This script performs data augmentation on images
def data_augmentation_layer(image):
    # Adds a series of data augmentation layers to the image
    data_augmentation = tf.keras.Sequential([
        layers.RandomContrast(0.5),
        # layers.RandomSaturation(0.7),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.025),
    ])
    return data_augmentation(image)

def main():
    data_dir = "Data" # Change this to your data folder name
    target_augmented = 412 # Change this to the number of augmented images you want

    for subfolder in os.listdir(data_dir):
        subfolder_path = os.path.join(data_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        # Ensure the subfolder contains images
        images = [f for f in os.listdir(subfolder_path) if f.lower().endswith('.jpg')]
        # num_existing = len(images)
        count = 0
        while count < target_augmented:
            for img_name in images:
                if count >= target_augmented:
                    break
                img_path = os.path.join(subfolder_path, img_name)
                
                # Load and preprocess the image for augmentation
                image = load_and_preprocess_image(img_path)
                augmented = data_augmentation_layer(image)
                
                # Ensure the augmented image is in the range [0, 1]
                augmented = tf.clip_by_value(augmented, 0.0, 1.0)
                
                # Save the augmented image
                save_name = f"aug_{count+1}_{img_name}"
                save_path = os.path.join(subfolder_path, save_name)
                tf.keras.utils.save_img(save_path, augmented)
                count += 1

if __name__ == "__main__":
    main()
