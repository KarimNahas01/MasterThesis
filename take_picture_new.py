import dataset_generator as dg
import annotation_tool as at
import dataset_tester as dt
import pyrealsense2 as rs
import tkinter as tk
import numpy as np
import threading
import json
import time
import cv2
import os

from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import LED, Button
from tkinter import simpledialog

# TODO: add these to a generic python file so that it can be imported. Also add commonly used methods there.
# TODO: make it possible to replace the transparent background with an image of your choice. To reperesent different environments
        
USING_REALSENSE_CAMERA = True
LEARNING_MODE = False

CONSTANTS = json.load(open('constants.json'))

OUTPUT_PORT = CONSTANTS['ports']['OUTPUT_PORT']
INPUT_PORT = CONSTANTS['ports']['INPUT_PORT']

RASPBERYY_PI_IP = CONSTANTS["ip_adress"]["raspberry_pi"]


IMAGE_RESOLUTION = CONSTANTS["image_props"]["resolution"]
STREAM_FRAMERATE = CONSTANTS["image_props"]["framerate"]

AUGMENT_VALUES = CONSTANTS["augment_values"]

VARIABLES_FILE = 'stored_variables.txt'

DIRECTORIES = CONSTANTS["directories"]

PARENT_FOLDER_NAME = DIRECTORIES["parent"]["folder_name"]

RGB_FOLDER_NAME = DIRECTORIES["rgb"]["folder_name"]
RGB_FILE_NAME = DIRECTORIES["rgb"]['file_name']

DEPTH_FOLDER_NAME = DIRECTORIES["depth"]["folder_name"]
DEPTH_FILE_NAME = DIRECTORIES["depth"]['file_name']

DEPTH_VALUES_FOLDER_NAME = DIRECTORIES["depth_values"]["folder_name"]
DEPTH_VALUES_FILE_NAME = DIRECTORIES["depth_values"]['file_name']

TRANSPARENT_FOLDER_NAME = DIRECTORIES["transparent"]["folder_name"]
TRANSPARENT_FILE_NAME = DIRECTORIES["transparent"]['file_name']

ANNOTATIONS_FOLDER_NAME = DIRECTORIES["annotations"]["folder_name"]
ANNOTATIONS_FILE_NAME = DIRECTORIES["annotations"]['file_name']

NORMALIZED_FOLDER_NAME = DIRECTORIES["normalized"]["folder_name"]
NORMALIZED_FILE_NAME = DIRECTORIES["normalized"]['file_name']

BLURRED_FOLDER_NAME = DIRECTORIES["blurred"]["folder_name"]
BLURRED_FILE_NAME = DIRECTORIES["blurred"]['file_name']

DATASET_STRUCTURE = CONSTANTS["dataset_structure"]
DATASET_PATH = f'{PARENT_FOLDER_NAME}/{DATASET_STRUCTURE["name"]}'
SUB_DIR_NAMES = DATASET_STRUCTURE['sub_dir_names']

folder_path = ""

try:
    with open(VARIABLES_FILE, 'r') as json_file:
        n_imgs = int(json_file.read())
    print("Number of images read from file:", n_imgs)

except FileNotFoundError:
    print(f"Error: File '{VARIABLES_FILE}' not found. Please use teach mode.")


root = tk.Tk()
root.withdraw()

factory = PiGPIOFactory(host=RASPBERYY_PI_IP)
move_robot = LED(OUTPUT_PORT, pin_factory=factory)
take_picture = Button(INPUT_PORT, pin_factory=factory)


if USING_REALSENSE_CAMERA:
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1], rs.format.bgr8, STREAM_FRAMERATE['color'])
    config.enable_stream(rs.stream.depth, IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1], rs.format.z16, STREAM_FRAMERATE['depth'])
    config.enable_stream(rs.stream.infrared, IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1], rs.format.y8, STREAM_FRAMERATE['infrared'])

    align_to = rs.stream.color # ----
    align = rs.align(align_to) # ----
    # pipe.start(config)

    # Start the RealSense pipeline
    profile = pipeline.start(config)
    device = profile.get_device()


rgb_folder_path, normalized_images_folder_path = "", ""

def check_elapsed_time():
    while True:
        move_robot.on()
        move_robot.off()
        if not take_picture.is_pressed:
            print("Demo has finished executing")
        os._exit(1)
        time.sleep(1)

demo_done_thread = threading.Thread(target=check_elapsed_time)

def main():
    global folder_path

    # for i in range(1,21):
    #     os.makedirs(f'C:/Users/karim/OneDrive - Chalmers/University/Complex Adaptive Systems MSc/Master thesis/git/img/test_connector/v.{i}.0', exist_ok=True)
    # return
    
    # connector_name =  input('\nInput name of connector: ').replace(" ", "_")
    connector_name = simpledialog.askstring(title="Connector name", prompt="Input name of connector:").replace(" ", "_") 

    # print("User input:", connector_name)

    setup_directories(connector_name, exclude_first=True)
    
    take_picture_logic(n_imgs)

    

                # print(depth_colormap.shape())
        # print(f'colormap: min={min(depth_colormap)}, \t max={max(depth_colormap)}')
        # print(f'image: min={np.min(depth_image)}, \t max={np.max(depth_image)}')
    # color_images = load_images_from_folder(rgb_folder_path)
    # depth_images = load_images_from_folder(depth_folder_path)
    

    
    # TEMP STUFF -------------
    '''
    folder_path = os.path.join('img', 'connector_more_poses_test')
    rgb_folder_path = f'{folder_path}/rgb_imgs'

    
    normalized_images_folder_path = f'{folder_path}/normalized_images'
    os.makedirs(normalized_images_folder_path, exist_ok=True)

    blurred_images = preprocessing()
    augment_images(load_images_from_folder(rgb_folder_path))
    '''

    # -----------------------

def take_picture_logic(n_imgs):
    global folder_path
    
    curr_img = 1
    print("Start the program on the robot \n")
    
    while curr_img <= n_imgs:
        if USING_REALSENSE_CAMERA:
            depth_colormap, depth_image, color_image, key = display_image(display_rgb=True, display_depth=True)
        # store_images(curr_img, depth_colormap, depth_image, color_image,
        #             rgb_folder_path, depth_folder_path, depth_values_folder_path)
        
        if not take_picture.is_pressed or key == ord('t'):
            # time.sleep(1)
            # TODO: FIX END CALL
            '''
            time.sleep(1)
            move_robot.on()
            move_robot.off()
            if take_picture.is_pressed:
                print('Execution finished')
                cv2.destroyAllWindows()
                if USING_REALSENSE_CAMERA: preprocessing()
                return
            '''

            print(f'Robot has reached position {curr_img}')
            if USING_REALSENSE_CAMERA:
                store_images({str(curr_img): color_image}, RGB_FOLDER_NAME, RGB_FILE_NAME)
                store_images({str(curr_img): depth_colormap}, DEPTH_FOLDER_NAME, DEPTH_FILE_NAME)
                store_images({str(curr_img): depth_image}, DEPTH_VALUES_FOLDER_NAME, DEPTH_VALUES_FILE_NAME, is_numpy=True)
                print(f'Depth and RGB image for pos {curr_img} has been stored. \n')

            else:
                print('Nothing has been stored. \n')
            curr_img += 1
            move_robot.on()
            time.sleep(0.5)
            move_robot.off()
    
    print("Execution finished")
    print("Starting the predicition")
    demo_done_thread.start()
    dt.predict()
    return

def learn_path():
    prompt = input("Are you sure you want to teach? (Y/N)")
    if prompt.lower() != 'y':
        return
    n_imgs = 0

    while True:

        if not take_picture.is_pressed:
            time.sleep(1)
            n_imgs += 1
            move_robot.on()
            time.sleep(0.5)
            move_robot.off()
            
        with open(VARIABLES_FILE, 'w') as f:
            f.write(str(n_imgs))

# Reading the number from the file

def preprocessing():
    print("Preprocessing started")

    color_images = load_images_from_folder(f"{folder_path}/{RGB_FOLDER_NAME}")
    blurred_images = apply_blur(color_images)


    idx = 3

    # cv2.imshow(f'Original image {idx}', color_images[idx])
    # cv2.imshow(f"Simple average blur | Gaussian blur for image {idx}",
    #             np.hstack((blurred_images[idx]['simple_average'], blurred_images[idx]['gaussian'])))
    # cv2.imshow(f"Median blur | Bilateral blur for image {idx}",
    #             np.hstack((blurred_images[idx]['median'], blurred_images[idx]['bilateral'])))
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()


    """ npy_normalized = normalize_and_store(color_images, normalized_images_folder_path)

    # print(np.shape(npy_normalized))

    is_normalized(color_images[1])
    is_normalized(npy_normalized[0]) """

    # exit()
    return blurred_images # and other preprocessed images

def augment_images(images):

    for idx, image in images.items():
        brightness_factor = np.random.randint(AUGMENT_VALUES["brightness_range"][0], [AUGMENT_VALUES["brightness_range"][1]])
        adjusted_image = np.clip(image.astype(int) + brightness_factor, 0, 255).astype(np.uint8)
        cv2.imshow(f"Original image | Brightness change (pos {idx})",
        np.hstack((image, adjusted_image)))
        cv2.waitKey(2000) 
        cv2.destroyAllWindows()
        
def apply_blur(images, simple_avg_kernal=(9,9), gaussian_kernal=(9,9), median_kernal=9,
                bilateral_consts={'d':15, 'sigmaColor':80, 'sigmaSpace':80}):
    blurred_images = {}
    for idx, img in images.items():
        simple_avg_blur = cv2.blur(img, simple_avg_kernal)
        gaussian_blur = cv2.GaussianBlur(img,gaussian_kernal,cv2.BORDER_DEFAULT)
        median_blur = cv2.medianBlur(img, median_kernal)
        bilateral_blur = cv2.bilateralFilter(img,bilateral_consts['d'],bilateral_consts['sigmaColor'],bilateral_consts['sigmaSpace'])

        blurred_images[idx] = {'simple_average': simple_avg_blur, 'gaussian': gaussian_blur, 
                            'median': median_blur, 'bilateral': bilateral_blur}
        
        store_images(blurred_images[idx], BLURRED_FOLDER_NAME, 'blur')
    
    return blurred_images

def get_current_version(connector_name):
    version = 1

    path = os.path.join(PARENT_FOLDER_NAME, f'{connector_name}_connector')

    if not os.path.exists(path):
        return 1

    existing_versions = sorted(os.listdir(path), key=lambda x: int(x.split('.')[1]))

    version = int(existing_versions[-1:][0].split('.')[1])

    return version+1

def setup_directories(connector_name, exclude_first=False):
    global folder_path

    version = get_current_version(connector_name)

    connector = f'{connector_name}_connector/v.{version}.0'
    folder_path = os.path.join(PARENT_FOLDER_NAME, connector)

    for idx, (_, directory) in enumerate(DIRECTORIES.items()):
        if not (exclude_first and idx == 0):
            path = os.path.join(folder_path, directory["folder_name"])
            os.makedirs(path, exist_ok=True)

    print(f'Initiated v.{version}.0')

    for _, value in list(DATASET_STRUCTURE.items())[2:]:
        path = f'{DATASET_PATH}/{value["name"]}'

        os.makedirs(f'{path}/{SUB_DIR_NAMES["images"]}', exist_ok=True)
        os.makedirs(f'{path}/{SUB_DIR_NAMES["labels"]}', exist_ok=True)
    
'''
def send_message(message, receiver_ip=RECEIVER_IP, receiver_port=RECEIVER_PORT ):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        sock.connect((receiver_ip, receiver_port))
        
        sock.sendall(message.encode())
        
        print(f'Message \'{message}\' sent successfully')
        
    except Exception as e:
        print(f"Error occurred while sending message: {e}")
        
    finally:
        sock.close()

def await_message_code(code):

    print('Awaiting message...')

    code_length = len(code)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = ('', LISTENING_PORT)
    sock.bind(server_address)

    sock.listen(1)
    while True:
        connection, _ = sock.accept()
        data = connection.recv(1024)

        if data:
            data_received = data.decode()
            code_received = data_received[:code_length]
            message_received = data_received[code_length:]

            print("Received:", data_received)
            
            if code_received == TERMINATE_CODE:
                print('Program terminated')
                exit()
            elif code_received == EXECUTION_FINISHED:
                connection.close()
                cv2.destroyAllWindows()
                preprocessing()
                main()

            elif code_received == code:
                print(f"Code {code_received} received.")
                print(f'Message received: {message_received}')
                break
    connection.close()
    return message_received
'''
def display_image(display_rgb=True, display_depth=True):
        frame = pipeline.wait_for_frames(timeout_ms=5000)
        aligned_frames = align.process(frame)

        frames = {
            'depth': aligned_frames.get_depth_frame(),
            'color': aligned_frames.get_color_frame(),
            'infrared': aligned_frames.get_infrared_frame()
        }
        # frames = {
        #     'depth': frame.get_depth_frame(), # ///
        #     'color': frame.get_color_frame(), # ///
        #     'infrared': frame.get_infrared_frame() # ///
        # }

        depth_image = np.asanyarray(frames['depth'].get_data())
        color_image = np.asanyarray(frames['color'].get_data())
        infrared_image = np.asanyarray(frames['infrared'].get_data())

        """ depth_stream = pipeline.get_active_profile().get_stream(rs.stream.depth)
        depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        depth_extrinsics = depth_stream.get_extrinsics_to(frame.get_profile().get_stream(rs.stream.color))


        # Access the rotation and translation components
        rotation = depth_extrinsics.rotation
        translation = depth_extrinsics.translation

        print("Rotation Matrix:")
        print(rotation)

        print("\nTranslation Vector:")
        print(translation)

        exit()

        extrinsics = ct.get_depth_to_color_extrinsics(frames)
        print(extrinsics)
        
        depth_image = cv2.flip(depth_image, 1)
        color_image = cv2.flip(color_image, 1)
        infrared_image = cv2.flip(infrared_image, 1)
        print('Before calulate values')
        bounding_box_points_color_image, length, width, height, point_cloud = calculate_values(calibration_output, depth_frame)
        print('After calulate values and before visualise measurements')
        visualise_measurements(color_image, bounding_box_points_color_image, length, width, height)
        visualise_measurements_multiple(frames_devices, bounding_box_points_color_image, length, width, height)
        print('After visualise measurements')
        """

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)

        # if display_rgb:      cv2.imshow('RGB image', color_image)
        # if display_depth:    cv2.imshow('Depth image', depth_colormap)
        # if display_infrared: cv2.imshow('Infrared image', infrared_image)

        combined_image = None

        if display_rgb:
            combined_image = color_image.copy() if combined_image is None else cv2.hconcat([combined_image, color_image])
        if display_depth:
            combined_image = depth_colormap.copy() if combined_image is None else cv2.hconcat([combined_image, depth_colormap])

        title_name = 'RGB and depth images' if display_rgb and display_depth else 'RGB image' if display_rgb else 'Depth image'

        if not combined_image is None:
            cv2.imshow(title_name, cv2.flip(combined_image, 0))
            key = cv2.waitKey(1)

        return depth_colormap, depth_image, color_image, key

def store_images(images, folder_name, file_name, is_numpy=False):

    # TODO: make a dictionary that includes the highest current stored image for each image type, rgb, depth, blurred etc. Add input of dict part or take from file_name

    for idx, (key, image) in enumerate(images.items()):
        suffix = key if key.isdigit() else f'{key}_{idx+1}'
        img_path = f'{folder_path}/{folder_name}/{file_name}_{suffix}'

        # print(img_path)

        if is_numpy: np.save(f'{img_path}.npy', image)
        else: cv2.imwrite(f'{img_path}.png', image)
        
        cv2.waitKey(100)

# def store_images(curr_img, depth_values, depth_image, color_image, 
#                  rgb_folder_path, depth_folder_path, depth_values_folder_path):

#     color_img_path = os.path.join(rgb_folder_path, f'color_img_pos_{curr_img}.png')
#     depth_img_path = os.path.join(depth_folder_path, f'depth_img_pos_{curr_img}.png')
#     depth_values_path = os.path.join(depth_values_folder_path, f'depth_values_pos_{curr_img}.npy')

#     depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)

#     cv2.imwrite(color_img_path, color_image)
#     cv2.imwrite(depth_img_path, depth_colormap)
#     np.save(depth_values_path, depth_values)

#     cv2.waitKey(500)

def load_images_from_folder(folder):
    # num_images = sum(1 for file in os.listdir(folder) if any(file.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"]))
    # images = np.empty((num_images,) + (RESOLUTION[0], RESOLUTION[1], 3), dtype=np.uint8) #creates empty array of images size (4, 640, 480, 3)
    images = {}
    print(folder)
    file_map = enumerate(sorted(os.listdir(folder), key=lambda x: int(''.join(filter(str.isdigit, x)))))

    for idx, filename in file_map:
        # Check if the file is an image (you can add more extensions if needed)
        if any(filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            # cv2.imshow("Inside load_images", img)
            # cv2.waitKey(0)
            if img is not None:
                images[idx+1] = img
                # images = np.append(images, img)
    return images

def normalize_and_store(images, normalized_images_folder_path):
    if not os.path.exists(normalized_images_folder_path):
        os.makedirs(normalized_images_folder_path)

    npy_images = []
    for idx, image in images.items():
        # Normalize pixel values to [0, 1]
        normalized_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        """ # Convert the normalized image to the appropriate data type (uint8) and scale pixel values to [0, 255]
        # normalized_image_uint8 = (normalized_image * 255).astype('uint8') """

        cv2.imshow(f'Normalized image {idx}', normalized_image)
        cv2.waitKey(2000)

        normalized_img_path = os.path.join(normalized_images_folder_path, f'normalized_img_pos_{idx}.npy')

        npy_image = np.array(normalized_image, dtype=np.float32)
        npy_images.append(npy_image)
        np.save(normalized_img_path, npy_image)

    return npy_images

def is_normalized(image):
    # Check if all pixel values are within the range [0, 1]
    min_val = np.min(image)
    max_val = np.max(image)
    print(f'Min: {min_val}, Max: {max_val}, is_normalized = {min_val >= 0 and max_val <= 1}')
    return min_val >= 0 and max_val <= 1

if __name__ == '__main__':
    if LEARNING_MODE:
        learn_path()
    else:
        main()
    time.sleep(3) # Necessary sleep before termination otherwise the Raspberry Pi complains!