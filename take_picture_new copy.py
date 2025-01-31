import dataset_tester as dt
import pyrealsense2 as rs
import tkinter as tk
import numpy as np
import threading
import time
import cv2
import sys
import os

from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import LED, Button
from tkinter import simpledialog


OUTPUT_PORT = 23    # AI 0
INPUT_PORT = 26     # AO 1

LISTENING_PORT = 30003
STARTUP_CODE = '011'
TAKE_PICTURE_CODE = '231'
TERMINATE_CODE = '404'
EXECUTION_FINISHED = '000'
START_BACKEND = '123'

# RASPBERYY_PI_IP = '192.168.2.218' # Ethernet
RASPBERYY_PI_IP = '192.168.50.161' # WiFI

USING_REALSENSE_CAMERA = True
LEARNING_MODE = False

# RESOLUTION = [1280, 720]
# FRAMERATE = {'color':15, 'depth':6, 'infrared':6}

RESOLUTION = [640, 480]
FRAMERATE = {'color':30, 'depth':30, 'infrared':30}

BRIGHTNESS_RANGE = (-120, 120)

# REPOSITORY =  {'parent': "img",
#                      'rgb': "rgb_images",
#                      'depth': "depth_images",
#                      'depth_values': "depth_values",
#                      'normalized': "normalized_images",
#                      'blurred': "blurred images"}

DIRECTORIES = {'parent':         "EWASS_demo_img",
               'rgb':            "rgb_images",
               'depth':          "depth_images",
               'depth_values':   "depth_values",
               'normalized':     "normalized_images",
               'blurred':        "blurred_images"}

VARIABLES_FILE = 'stored_variables.txt'

folder_path = ""


try:
    with open(VARIABLES_FILE, 'r') as f:
        n_imgs = int(f.read())
    print("Number of images read from file:", n_imgs)

except FileNotFoundError:
    print(f"Error: File '{VARIABLES_FILE}' not found. Please use teach mode.")


root = tk.Tk()
root.withdraw()  # Hide the main window

factory = PiGPIOFactory(host=RASPBERYY_PI_IP)
move_robot = LED(OUTPUT_PORT, pin_factory=factory)
take_picture = Button(INPUT_PORT, pin_factory=factory)


if USING_REALSENSE_CAMERA:
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.bgr8, FRAMERATE['color'])
    config.enable_stream(rs.stream.depth, RESOLUTION[0], RESOLUTION[1], rs.format.z16, FRAMERATE['depth'])
    config.enable_stream(rs.stream.infrared, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FRAMERATE['infrared'])

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
    connector_name = "demo_tester" #simpledialog.askstring(title="Connector name", prompt="Input name of connector:").replace(" ", "_")

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
    
    curr_img = 0
    print("Start the program on the robot \n")
    
    while curr_img <= n_imgs:
        if USING_REALSENSE_CAMERA:
            depth_colormap, depth_image, color_image = display_image(display_rgb=True, display_depth=True)
        # store_images(curr_img, depth_colormap, depth_image, color_image,
        #             rgb_folder_path, depth_folder_path, depth_values_folder_path)
        
        if not take_picture.is_pressed:
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
                store_images({str(curr_img): color_image}, DIRECTORIES['rgb'], 'color_img')
                store_images({str(curr_img): depth_colormap}, DIRECTORIES['depth'], 'depth_img')
                store_images({str(curr_img): depth_image}, DIRECTORIES['depth_values'], 'depth_values', is_numpy=True)
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

    color_images = load_images_from_folder(f"{folder_path}/{DIRECTORIES['rgb']}")
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
        brightness_factor = np.random.randint(BRIGHTNESS_RANGE[0], [BRIGHTNESS_RANGE[1]])
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
        
        store_images(blurred_images[idx], DIRECTORIES['blurred'], 'blur')
    
    return blurred_images

def get_current_version(connector_name):
    version = 1

    path = os.path.join(DIRECTORIES['parent'], f'{connector_name}_connector')

    if not os.path.exists(path):
        return 1

    existing_versions = sorted(os.listdir(path), key=lambda x: int(x.split('.')[1]))

    version = int(existing_versions[-1:][0].split('.')[1])

    return version+1

def setup_directories(connector_name, exclude_first=False):
    global folder_path

    version = get_current_version(connector_name)

    connector = f'{connector_name}_connector/v.{version}.0'
    folder_path = os.path.join(DIRECTORIES['parent'], connector)

    for idx, (_, directory) in enumerate(DIRECTORIES.items()):
        if not (exclude_first and idx == 0):
            path = os.path.join(folder_path, directory)
            os.makedirs(path, exist_ok=True)

    print(f'Initiated v.{version}.0')
    
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

        # Check which images to display and concatenate them horizontally
        if display_rgb:
            combined_image = color_image.copy() if combined_image is None else cv2.hconcat([combined_image, color_image])
        if display_depth:
            combined_image = depth_colormap.copy() if combined_image is None else cv2.hconcat([combined_image, depth_colormap])

        title_name = 'RGB and depth images' if display_rgb and display_depth else 'RGB image' if display_rgb else 'Depth image'

        # Display the combined image
        if not combined_image is None:
            cv2.imshow(title_name, cv2.flip(combined_image, 0))
            cv2.waitKey(1)

        return depth_colormap, depth_image, color_image

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