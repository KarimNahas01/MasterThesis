import pyrealsense2 as rs
import numpy as np
import threading
import json
import time
import yaml
import cv2
import sys
import os

import matplotlib.pyplot as plt

USING_REALSENSE_CAMERA = False
DEPTH_RANGE = 354
WHITE_RANGE = (90, 255)
REMOVE_BACKGROUND_TYPE = 'color' # REMOVE_BACKGROUND_TYPE in ['both', 'color', 'depth']


IMAGE_RESOLUTION = [640, 480]
FRAMERATE = {"color":30, "depth":30, "infrared":30}

CONSTANTS = json.load(open('constants.json'))
CLASSES_FILE = 'classes.json'

DIRECTORIES = CONSTANTS["directories"]

PARENT_FOLDER_NAME = DIRECTORIES["parent"]['folder_name']

RGB_FOLDER_NAME = DIRECTORIES["rgb"]['folder_name']
RGB_FILE_NAME = DIRECTORIES["rgb"]['file_name']

DEPTH_FOLDER_NAME = DIRECTORIES["depth"]['folder_name']
DEPTH_FILE_NAME = DIRECTORIES["depth"]['file_name']

DEPTH_VALUES_FOLDER_NAME = DIRECTORIES["depth_values"]['folder_name']
DEPTH_VALUES_FILE_NAME = DIRECTORIES["depth_values"]['file_name']

TRANSPARENT_FOLDER_NAME = DIRECTORIES["transparent"]['folder_name']
TRANSPARENT_FILE_NAME = DIRECTORIES["transparent"]['file_name']

ANNOTATIONS_FOLDER_NAME = DIRECTORIES["annotations"]['folder_name']
ANNOTATIONS_FILE_NAME = DIRECTORIES["annotations"]['file_name']

ANNOTATED_IMAGES_FOLDER_NAME = DIRECTORIES["annotated_images"]['folder_name']
ANNOTATED_IMAGES_FILE_NAME = DIRECTORIES["annotated_images"]['file_name']

DATASET_STRUCTURE = CONSTANTS["dataset_structure"]
DATASET_PATH = f'{PARENT_FOLDER_NAME}/{DATASET_STRUCTURE["name"]}'
DATA_YAML_FILE = f'{DATASET_PATH}/data.yaml'

if USING_REALSENSE_CAMERA:
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1], rs.format.bgr8, FRAMERATE['color'])
    config.enable_stream(rs.stream.depth, IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1], rs.format.z16, FRAMERATE['depth'])
    config.enable_stream(rs.stream.infrared, IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1], rs.format.y8, FRAMERATE['infrared'])

    align_to = rs.stream.color # ----
    align = rs.align(align_to) # ----
    # pipe.start(config)

    # Start the RealSense pipeline
    profile = pipeline.start(config)
    device = profile.get_device()

def main():

    if not USING_REALSENSE_CAMERA:
        # annotate_data('tmp/dark_gray_large_1_connector/v.1.0', connector_name='dark_gray_large')
        # annotate_data('tmp/pictures_for_demo_connector/v.2.0', connector_name='light_gray_large')
        # annotate_data('annotation_testing_img/large_light_gray_connector/v.1.0', connector_name='light_gray_large')
        # annotate_data('EWASS_demo_img/light_gray_big_1_connector/v.1.0', connector_name='light_gray_large')
        file_map = os.listdir(PARENT_FOLDER_NAME)

        for folder_name in file_map:
            if folder_name == DATASET_PATH.split('/')[1]:
                continue
            versions = os.listdir(f'{PARENT_FOLDER_NAME}/{folder_name}')
            max_version = max(versions, key=lambda x: float(x.split('.')[1]))
            path = f'{PARENT_FOLDER_NAME}/{folder_name}/{max_version}'
            connector_name = folder_name.replace('_connector','')
            annotate_data(path, connector_name=connector_name)

    else:
        while True:
            annotate_using_camera(display_transparent=True, display_annotated=True)

def annotate_data(folder_path, connector_name):
    rgb_images = load_images_from_folder(f'{folder_path}/{RGB_FOLDER_NAME}')
    depth_images = load_images_from_folder(f'{folder_path}/{DEPTH_FOLDER_NAME}')
    depth_values = load_images_from_folder(f'{folder_path}/{DEPTH_VALUES_FOLDER_NAME}')

    connector_class = get_connector_class(connector_name)

    for idx in range(len(rgb_images.items())):
        depth_value = list(depth_values.items())[idx][1]
        rgb = list(rgb_images.items())[idx][1]
        depth = list(depth_images.items())[idx][1]
                    
        # depth_jet = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.5), cv2.COLORMAP_JET)
        transparent_img = remove_background(depth_value, rgb, idx+1, folder_path, verbose=True, save_img=True)
        if transparent_img is not None:
            annotations = get_annotations_from_contours(transparent_img, path=f'{folder_path}/{RGB_FOLDER_NAME}/{RGB_FILE_NAME}_{idx+1}.png', connector_class=connector_class)
            annotated_img = draw_bounding_box(rgb, annotations, connector_name, verbose=True)
            
            store_annotations(folder_path, annotations, annotated_img, idx+1)

def get_connector_class(connector_name):

    # Read the existing YAML data
    with open(DATA_YAML_FILE, 'r') as file:
        data = yaml.safe_load(file)

    classes = data["names"]

    for key, value in classes.items():
        if value == connector_name:
            return key

    """
    classes = json.load(open('classes.json'))

    for key, value in classes["names"].items():
        if value == connector_name:
            return key

    new_key = str(len(classes["names"]))
    classes["names"][new_key] = connector_name

    with open(CLASSES_FILE, 'w') as f:
        json.dump(classes, f, indent=4)

    return new_key
    """

def remove_background(depth_value, rgb_image, idx, folder_path=None, verbose=False, save_img=False):

    rgbd_array = np.concatenate((rgb_image, np.expand_dims(depth_value, axis=2)), axis=2)
    rgbd_array[(rgbd_array[..., 3] < 10) | (rgbd_array[..., 3] > DEPTH_RANGE), :3] = [255, 255, 255]
    new_rgb_image = rgbd_array[:, :, :3].astype(np.uint8)
    rgba_image = cv2.cvtColor(new_rgb_image, cv2.COLOR_RGB2RGBA)
    rgba_image[np.all(rgba_image[:, :, :3] == [255, 255, 255], axis=-1)] = [255, 255, 255, 0]
    depth_result = rgba_image.copy()#[:, :, :3].astype(np.uint8)

    # HSV instead of RGB:
    """ 
        # hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        # lower_white = np.array([42, 12, 88]) #96968F
        # upper_white = np.array([96, 50, 250]) #545855
        # mask = cv2.inRange(hsv, lower_white, upper_white)
        # mask = cv2.bitwise_not(mask)
        # colors_image = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)
        """

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, WHITE_RANGE[0], WHITE_RANGE[1], cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask)
    colors_image = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)
    rgba_image = cv2.cvtColor(colors_image, cv2.COLOR_RGB2RGBA)
    rgba_image[np.all(rgba_image[:, :, :3] == [0, 0, 0], axis=-1)] = [0, 0, 0, 0]
    colors_result = rgba_image.copy()#[:, :, :3].astype(np.uint8)
        
    background_type_mapping = {
        'both': (cv2.hconcat([depth_result, colors_result]), "Using depth vs using colors"),
        'color': (colors_result, "Using colors"),
        'depth': (depth_result, "Using depth values")
    }

    transparent_image, title = background_type_mapping.get(REMOVE_BACKGROUND_TYPE, (None, "Unknown Background Type"))
    
    if transparent_image is None:
        print(title) 
        return

    if verbose:
        cv2.imshow(title, transparent_image)
        cv2.waitKey(500)

    # Save image
    if save_img:
        os.makedirs(f'{folder_path}/{TRANSPARENT_FOLDER_NAME}', exist_ok=True) # Not needed when ran from take_picture
    
        cv2.imwrite(f'{folder_path}/{TRANSPARENT_FOLDER_NAME}/{TRANSPARENT_FILE_NAME}_{idx}.png', transparent_image)

    if REMOVE_BACKGROUND_TYPE != 'both': return transparent_image

def convert_to_yolo(bbox, image_width, image_height):
    x_min, y_min, x_max, y_max, class_label = bbox
    x_center = (x_min + x_max) / (2 * image_width)
    y_center = (y_min + y_max) / (2 * image_height)
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height

    return class_label, x_center, y_center, width, height

def store_annotations(folder_path, annotations, annotated_img, idx):
    os.makedirs(f'{folder_path}/{ANNOTATIONS_FOLDER_NAME}', exist_ok=True) # Not needed when ran from take_picture
    os.makedirs(f'{folder_path}/{ANNOTATED_IMAGES_FOLDER_NAME}', exist_ok=True) # Not needed when ran from take_picture

    with open(f'{folder_path}/{ANNOTATIONS_FOLDER_NAME}/{ANNOTATIONS_FILE_NAME}_{idx}.txt', "w") as yolo_file:
        for annotation in annotations:
            image_path, x_min, y_min, x_max, y_max, class_label = annotation
            image = cv2.imread(image_path)
            image_height, image_width, _ = image.shape
            yolo_annotation = convert_to_yolo((x_min, y_min, x_max, y_max, class_label), image_width, image_height)
            yolo_file.write(" ".join([str(item) for item in yolo_annotation]) + "\n")

    cv2.imwrite(f'{folder_path}/{ANNOTATED_IMAGES_FOLDER_NAME}/{ANNOTATED_IMAGES_FILE_NAME}_{idx}.jpg', annotated_img)

def draw_bounding_box(image, annotations, connector_name, verbose=False):
    for annotation in annotations:
        (_, x_min, y_min, x_max, y_max, _) = annotation
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        cv2.rectangle(image, (x_min, y_min - 20), (x_min + len(connector_name) * 9, y_min), (255, 0, 0), cv2.FILLED)
        
        cv2.putText(image, connector_name.replace('_',' '), (x_min+5, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if verbose:
        cv2.imshow("Annotated image", image)
        cv2.waitKey(500)
    return image

def get_annotations_from_contours(image, path, connector_class):
    _, _, _, alpha = cv2.split(image)

    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotations = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if w*h < 500:
            continue

        annotations.append((path, x, y, x + w, y + h, connector_class))

    return annotations

# TODO: remove when done replacing with annotate_data
"""
def extract_connector(folder_path, save_img=False):
    rgb_images = load_images_from_folder(f'{folder_path}/{RGB_FOLDER_NAME}')
    depth_images = load_images_from_folder(f'{folder_path}/{DEPTH_FOLDER_NAME}')
    depth_values = load_images_from_folder(f'{folder_path}/{DEPTH_VALUES_FILE_NAME}')
    for idx in range(len(rgb_images.items())):
        depth_value = list(depth_values.items())[idx][1]
        rgb = list(rgb_images.items())[idx][1]
        depth = list(depth_images.items())[idx][1]
                    
        depth_jet = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.5), cv2.COLORMAP_JET)
        transparent_img = remove_background(depth_value, rgb, verbose=True)
        
        os.makedirs('tmp/transparent_images', exist_ok=True)
        if save_img:
           cv2.imwrite(f'tmp/transparent_images/transparent_img_{idx+1}.png', transparent_img)

        bbox = draw_bounding_box(transparent_img, path=f'{folder_path}/rgb_images_color_img_{idx+1}')

        # print(rgbd_array[200][300])

        # print(np.shape(depth))
        # print(np.shape(depth_jet))
        # depth_jet_2d = depth_jet.reshape((480, 640, 3))

        # rgb[depth_jet_2d >= 150] = [255, 255, 255]
        # rgb[depth_jet >= 150] = [255, 255, 255]
        # tmp_idx = 56
        # x = int(tmp_idx / 640)
        # y = tmp_idx % 480
        # print(rgb[x][y])
        # cv2.rectangle(rgb, (300, 100), (475, 300), (0, 255, 0), 2)
        # cv2.rectangle(depth, (300, 100), (475, 300), (0, 255, 0), 2)
        # plt.imshow(depth_value)
        # plt.show()
        # disp_img = cv2.hconcat([depth, rgb_array])
"""


def load_images_from_folder(folder, verbose=False):
    
    images = {}
    file_map = enumerate(sorted(os.listdir(folder), key=lambda x: int(''.join(filter(str.isdigit, x)))))

    for idx, filename in file_map:
        if any(filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)

            if img is not None:
                images[idx+1] = img
            if verbose:
                cv2.imshow("Inside load_images", img)
                cv2.waitKey(500)
        if filename.lower().endswith(".npy"):
            img_path = os.path.join(folder, filename)
            img = np.load(img_path)
            images[idx+1] = img
    return images

def annotate_using_camera(display_transparent=True, display_annotated=True):
        frame = pipeline.wait_for_frames(timeout_ms=5000)
        aligned_frames = align.process(frame)

        frames = {
            'depth': aligned_frames.get_depth_frame(),
            'color': aligned_frames.get_color_frame()
        }

        depth_image = np.asanyarray(frames['depth'].get_data())
        color_image = np.asanyarray(frames['color'].get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)

        # plt.imshow(depth_image)
        # plt.show()
        # return

        combined_image = None

        transparent_img = remove_background(depth_image, color_image, idx=0, verbose=False, save_img=False)

        if transparent_img is None: return
        
        # transparent_img = transparent_img[:, :, :3].astype(np.uint8)

        center_x = transparent_img.shape[1] // 2
        center_y = transparent_img.shape[0] // 2
        arm_length = 25
        # cv2.line(color_image, (center_x - arm_length, center_y), (center_x + arm_length, center_y), (0, 0, 255), 1)
        # cv2.line(color_image, (center_x, center_y - arm_length), (center_x, center_y + arm_length), (0, 0, 255), 1)

        depth_colormap = transparent_img.copy()

    
        annotations = get_annotations_from_contours(transparent_img, path=None, connector_class="connector")
        annotated_img = draw_bounding_box(color_image, annotations, connector_name="connector", verbose=False)
        transparent_img = transparent_img[:, :, :3].astype(np.uint8)

        if display_transparent:
            combined_image = transparent_img.copy() if combined_image is None else cv2.hconcat([combined_image, transparent_img])
        if display_annotated:
            combined_image = annotated_img.copy() if combined_image is None else cv2.hconcat([combined_image, annotated_img])

        title_name = 'Transparent and annotated images' if display_transparent and display_annotated else 'Transparent image' if display_transparent else 'Annotated image'

        if not combined_image is None:
            # cv2.imshow(title_name, cv2.flip(combined_image, 0))
            cv2.imshow(title_name, combined_image)
            cv2.waitKey(1)

        return depth_colormap, depth_image, color_image


if __name__ == '__main__':
    main()