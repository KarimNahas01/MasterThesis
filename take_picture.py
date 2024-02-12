import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import socket

STARTUP_CODE = '011'
TAKE_PICTURE_CODE = '231'
TERMINATE_CODE = '404'

pipeline = rs.pipeline()
config = rs.config()

# RESOLUTION = [640, 480]
RESOLUTION = [1280, 720]
FRAMERATE = {'color':15, 'depth':6, 'infrared':6}

config.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.bgr8, FRAMERATE['color'])
config.enable_stream(rs.stream.depth, RESOLUTION[0], RESOLUTION[1], rs.format.z16, FRAMERATE['depth'])
config.enable_stream(rs.stream.infrared, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FRAMERATE['infrared'])

align_to = rs.stream.color # ----
align = rs.align(align_to) # ----
# pipe.start(config)

# Start the RealSense pipeline
profile = pipeline.start(config)
device = profile.get_device()

def main():

    connector = await_message_code(STARTUP_CODE)
    rgb_folder_path, depth_folder_path, depth_values_folder_path = setup_directories(connector)
    
    while True:
        curr_img = await_message_code(TAKE_PICTURE_CODE)
        depth_colormap, depth_image, color_image = display_image(display_depth=False, display_infrared=False)
        store_images(curr_img, depth_colormap, depth_image, color_image,
                    rgb_folder_path, depth_folder_path, depth_values_folder_path)



def setup_directories(connector, images_folder_name='img'):
    folder_path = os.path.join(images_folder_name, f'connector_{connector}')

    rgb_folder_path = f'{folder_path}/rgb_imgs'
    depth_folder_path = f'{folder_path}/depth_imgs'
    depth_values_folder_path = f'{folder_path}/depth_values'

    os.makedirs(images_folder_name, exist_ok=True)
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(rgb_folder_path, exist_ok=True)
    os.makedirs(depth_folder_path, exist_ok=True)
    os.makedirs(depth_values_folder_path, exist_ok=True)

    return rgb_folder_path, depth_folder_path, depth_values_folder_path

def await_message_code(code):

    code_length = len(code)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = ('', 30003)  # Example listening port
    sock.bind(server_address)

    sock.listen(1)
    while True:
        connection, _ = sock.accept()
        data = connection.recv(1024)
        display_image(display_depth=False, display_infrared=False)

        if data:
            data_received = data.decode()
            code_received = data_received[:code_length]
            message_received = data_received[code_length:]

            print("Received:", data_received)
            if code_received == TERMINATE_CODE:
                print('Program terminated')
                exit()
            if code_received == code:
                print(f"Code {code_received} received.")
                print(f'Message received: {message_received}')
                break
    connection.close()
    return message_received

def display_image(display_rgb=True, display_depth=True, display_infrared=True):
        frame = pipeline.wait_for_frames(timeout_ms=5000)
        # aligned_frames = align.process(frame)
        frames = {
            'depth': frame.get_depth_frame(), # ///
            'color': frame.get_color_frame(), # ///
            'infrared': frame.get_infrared_frame() # ///
        }

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

        if display_rgb:      cv2.imshow('RGB image', color_image)
        if display_depth:    cv2.imshow('Depth image', depth_colormap)
        if display_infrared: cv2.imshow('Infrared image', infrared_image)

        return depth_colormap, depth_image, color_image

def store_images(curr_img, depth_values, depth_image, color_image, 
                 rgb_folder_path, depth_folder_path, depth_values_folder_path):

    color_img_path = os.path.join(rgb_folder_path, f'color_img_pos_{curr_img}.png')
    depth_img_path = os.path.join(depth_folder_path, f'depth_img_pos_{curr_img}.png')
    depth_values_path = os.path.join(depth_values_folder_path, f'depth_values_pos_{curr_img}.npy')

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)

    cv2.imwrite(color_img_path, color_image)
    cv2.imwrite(depth_img_path, depth_colormap)
    np.save(depth_values_path, depth_values)

    cv2.waitKey(500)

if __name__ == '__main__':
    main()