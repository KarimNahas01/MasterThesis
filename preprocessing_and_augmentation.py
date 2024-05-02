## PREPROCESSING & AUGMENTATION ##

import take_picture_new as tpk
import pyrealsense2 as rs
import numpy as np
import json
import cv2
import os
import json

CONSTANTS = json.load(open('constants.json'))
brightness_values = CONSTANTS["augment_values"]["brightness_range"]

def normalize_and_store(images, normalized_images_folder_path):
    if not os.path.exists(normalized_images_folder_path):
        os.makedirs(normalized_images_folder_path)

    npy_images = []
    for idx, image in images.items():
        # Normalize pixel values to [0, 1]
        normalized_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        """ # Convert the normalized image to the appropriate data type (uint8) and scale pixel values to [0, 255]
        # normalized_image_uint8 = (normalized_image * 255).astype('uint8') """

        # cv2.imshow(f'Normalized image {idx}', normalized_image)
        # cv2.waitKey(500)

        normalized_img_path = os.path.join(normalized_images_folder_path, f'normalized_img_{idx}.npy')

        npy_image = np.array(normalized_image, dtype=np.float32)
        npy_images.append(npy_image)
        np.save(normalized_img_path, npy_image)

    return npy_images

def resize_and_store(images, resized_image_folder_path):
    """
    Resize images to 640x480 if they are not already that size.
    
    Args:
    - images: List of images as NumPy arrays
    
    Returns:
    - List of resized images as NumPy arrays
    """
    resized_images = []
    for idx, image in images.items():
        # Get current image dimensions
        height, width, _ = image.shape

        resized_img_path = os.path.join(resized_image_folder_path, f'resized_img_{idx}.npy')
        
        # Check if the image dimensions match the desired size (640x480)
        if height != 480 or width != 640:
            # Resize the image to 640x480
            resized_image = cv2.resize(image, (640, 480))
            resized_images.append(resized_image)
            np.save(resized_img_path, resized_image)
        else:
            resized_images.append(image)
            np.save(resized_img_path, image)
        
    return resized_images

def motion_blur_and_store(images, augmented_data_folder_path ,kernel_size, angle):
    """
    Apply motion blur to an array of images.
    
    Args:
    - images: Array of input images as NumPy arrays
    - kernel_size: Size of the motion blur kernel
    - angle: Angle of motion blur (in degrees)
    
    Returns:
    - Array of images with motion blur applied as NumPy arrays
    """
    blurred_images = []
    for idx, image in images.items():

        blurred_img_path = os.path.join(augmented_data_folder_path, f'blurred_img_{idx}.npy')

        kernel = np.zeros((kernel_size, kernel_size))
        angle_rad = np.deg2rad(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        center = kernel_size // 2
        for i in range(kernel_size):
            x = i - center
            for j in range(kernel_size):
                y = j - center
                if np.abs(x * cos_angle + y * sin_angle) < 1:
                    kernel[i, j] = 1 / kernel_size
        blurred_image = cv2.filter2D(image, -1, kernel)
        # cv2.imshow(f'Blurred image', blurred_image)
        # cv2.waitKey(0)
        blurred_images.append(blurred_image)
        np.save(augmented_data_folder_path, image)

    return blurred_images

def add_gaussian_noise(images, augmented_data_folder_path, mean=0, stddev=10):
    """
    Add Gaussian noise to the given image.
    
    Args:
    - image: Input image as a NumPy array
    - mean: Mean of the Gaussian distribution (default is 0)
    - stddev: Standard deviation of the Gaussian distribution (default is 10)
    
    Returns:
    - Image with added Gaussian noise as a NumPy array
    """
    noisy_images = []
    for idx, image in images.items():

        noisy_img_path = os.path.join(augmented_data_folder_path, f'noisy_img_{idx}.npy')
        # Generate Gaussian noise with the same shape as the input image
        noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
        
        # Add the noise to the image
        noisy_image = cv2.add(image, noise)

        # cv2.imshow(f'Noisy image', noisy_image)
        # cv2.waitKey(0)

        np.save(augmented_data_folder_path, image)
        
    return noisy_images

def add_salt_and_pepper_noise(image, augmented_data_folder_path, salt_prob=0.01, pepper_prob=0.01):
    """
    Add salt and pepper noise to the given image.
    
    Args:
    - image: Input image as a NumPy array
    - salt_prob: Probability of adding salt noise to each pixel (default is 0.01)
    - pepper_prob: Probability of adding pepper noise to each pixel (default is 0.01)
    
    Returns:
    - Image with added salt and pepper noise as a NumPy array
    """
    noisy_images = []
    for idx, image in images.items():

        noisy_img_path = os.path.join(augmented_data_folder_path, f'noisy_img_{idx}.npy')
        noisy_image = np.copy(image)
        
        # Generate random matrix with the same shape as the image
        salt_mask = np.random.rand(*image.shape[:2])
        pepper_mask = np.random.rand(*image.shape[:2])
        
        # Add salt noise
        noisy_image[salt_mask < salt_prob] = 255
        
        # Add pepper noise
        noisy_image[pepper_mask < pepper_prob] = 0

        # cv2.imshow(f'Noisy image', noisy_image)
        # cv2.waitKey(0)

        np.save(augmented_data_folder_path, noisy_image)
    
    return noisy_image

def apply_random_brightness(images, augmented_data_folder_path, min_brightness=brightness_values[0], max_brightness=brightness_values[1]):
    """
    Apply random brightness adjustment to each image in the array.
    
    Args:
    - images: List of input images as NumPy arrays
    - min_brightness: Minimum brightness adjustment value (default is -50)
    - max_brightness: Maximum brightness adjustment value (default is 50)
    
    Returns:
    - List of images with random brightness adjustments as NumPy arrays
    """
    brightened_images = []
    for idx, image in images.items():

        noisy_img_path = os.path.join(augmented_data_folder_path, f'noisy_img_{idx}.npy')
        # Generate a random brightness adjustment value
        brightness_delta = np.random.randint(min_brightness, max_brightness + 1)
        
        # Apply brightness adjustment
        brightened_image = np.clip(image.astype(int) + brightness_delta, 0, 255).astype(np.uint8)

        cv2.imshow(f'Brightened image', brightened_image)
        cv2.waitKey(0)

        brightened_images.append(brightened_image)

        np.save(augmented_data_folder_path, brightened_image)

    return brightened_images


## PREPROCESSING ##

# Normalizing #
#Load images
images = tpk.load_images_from_folder('test2/')
print(type(images))
#Normalize and store
normalize_and_store(images,'test/')

#Check size 640x480
resized_images = resize_and_store(images, 'test/')

## AUGMENTATION ##

# Motion Blur
blurred_images = motion_blur_and_store(images, 'test/', kernel_size=3, angle=30)
# Add noise
gaussian_noisy_images = add_gaussian_noise(images, 'test/', mean=20, stddev=5)
salt_and_pepper_noisy_images = add_salt_and_pepper_noise(images, 'test/', salt_prob=0.002, pepper_prob=0.002)
# Vary brightness
brightened_images = apply_random_brightness(images, 'test/', min_brightness=brightness_values[0], max_brightness=brightness_values[1])


# def preprocessing():
#     print("Preprocessing started")
    
#     #Load images
#     color_images = load_images_from_folder(f"{folder_path}/{RGB_FOLDER_NAME}")
#     #Apply blur
#     blurred_images = apply_blur(color_images)


#     idx = 3

#     # cv2.imshow(f'Original image {idx}', color_images[idx])
#     # cv2.imshow(f"Simple average blur | Gaussian blur for image {idx}",
#     #             np.hstack((blurred_images[idx]['simple_average'], blurred_images[idx]['gaussian'])))
#     # cv2.imshow(f"Median blur | Bilateral blur for image {idx}",
#     #             np.hstack((blurred_images[idx]['median'], blurred_images[idx]['bilateral'])))
#     # cv2.waitKey(0) 
#     # cv2.destroyAllWindows()


#     """ npy_normalized = normalize_and_store(color_images, normalized_images_folder_path)

#     # print(np.shape(npy_normalized))

#     is_normalized(color_images[1])
#     is_normalized(npy_normalized[0]) """

#     # exit()
#     return blurred_images # and other preprocessed images

# def augment_images(images):

#     for idx, image in images.items():
#         brightness_factor = np.random.randint(AUGMENT_VALUES["brightness_range"][0], [AUGMENT_VALUES["brightness_range"][1]])
#         adjusted_image = np.clip(image.astype(int) + brightness_factor, 0, 255).astype(np.uint8)
#         cv2.imshow(f"Original image | Brightness change (pos {idx})",
#         np.hstack((image, adjusted_image)))
#         cv2.waitKey(2000) 
#         cv2.destroyAllWindows()
        
# def apply_blur(images, simple_avg_kernal=(9,9), gaussian_kernal=(9,9), median_kernal=9, motion_kernal=9, motion_angle=45,
#                 bilateral_consts={'d':15, 'sigmaColor':80, 'sigmaSpace':80}):
#     blurred_images = {}
#     for idx, img in images.items():
#         simple_avg_blur = cv2.blur(img, simple_avg_kernal)
#         gaussian_blur = cv2.GaussianBlur(img,gaussian_kernal,cv2.BORDER_DEFAULT)
#         median_blur = cv2.medianBlur(img, median_kernal)
#         bilateral_blur = cv2.bilateralFilter(img,bilateral_consts['d'],bilateral_consts['sigmaColor'],bilateral_consts['sigmaSpace'])
#         motion_blur = motion_blur(img, motion_kernal, motion_angle)

#         blurred_images[idx] = {'simple_average': simple_avg_blur, 'gaussian': gaussian_blur, 
#                             'median': median_blur, 'bilateral': bilateral_blur, 'motion': motion_blur}
        
#         store_images(blurred_images[idx], BLURRED_FOLDER_NAME, 'blur')
    
#     return blurred_images

