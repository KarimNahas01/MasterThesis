import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

import calibration_tools as ct
# from box_dimensioner_multicam.box_dimensioner_multicam_demo import run_calibration#, calculate_values, visualise_measurements
# from box_dimensioner_multicam.measurement_task import visualise_measurements as visualise_measurements_multiple

def main():

    # calibration_output = run_calibration()

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)

    align_to = rs.stream.color # ----
    align = rs.align(align_to) # ----

    # pipe.start(config)

    # Start the RealSense pipeline
    profile = pipeline.start(config)
    device = profile.get_device()

    # Load the JSON settings file
    # device = ct.load_settings_json(device, "box_dimensioner_multicam/HighResHighAccuracyPreset.json")

    # Get the depth-to-color extrinsics
    # extrinsics_device = ct.get_depth_to_color_extrinsics(device)
    # print(extrinsics_device)

    # Perform your measurement and display logic here

    n_imgs = 5
    curr_img = 1
    images_folder_name = 'img'
    connector = "robot_testing_2" # input('Input name of connector: ').replace(" ", "_")
    message_sent = False
    folder_path = os.path.join(images_folder_name, f'connector_{connector}')
    timer = time.time()
    move_robot = True
    robot_moving = False

    rgb_folder_path = f'{folder_path}/rgb_imgs'
    depth_folder_path = f'{folder_path}/depth_imgs'
    depth_values_folder_path = f'{folder_path}/depth_values'

    os.makedirs(images_folder_name, exist_ok=True)
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(rgb_folder_path, exist_ok=True)
    os.makedirs(depth_folder_path, exist_ok=True)
    os.makedirs(depth_values_folder_path, exist_ok=True)

    # data = np.load('img/connector_testing_123/depth_values/depth_values_pos_1.npy')
    # print(data[1:10, 1:10])
    # return
    while True:
        time_elapsed = time.time() - timer

        frame = pipeline.wait_for_frames()
        # aligned_frames = align.process(frame)
        frames = {
            'depth': frame.get_depth_frame(), # ///
            'color': frame.get_color_frame(), # ///
            'infrared': frame.get_infrared_frame() # ///
        }

        depth_image = np.asanyarray(frames['depth'].get_data())
        color_image = np.asanyarray(frames['color'].get_data())
        infrared_image = np.asanyarray(frames['infrared'].get_data())

        # depth_stream = pipeline.get_active_profile().get_stream(rs.stream.depth)
        # depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        # depth_extrinsics = depth_stream.get_extrinsics_to(frame.get_profile().get_stream(rs.stream.color))


        # # Access the rotation and translation components
        # rotation = depth_extrinsics.rotation
        # translation = depth_extrinsics.translation

        # print("Rotation Matrix:")
        # print(rotation)

        # print("\nTranslation Vector:")
        # print(translation)

        # exit()

        # extrinsics = ct.get_depth_to_color_extrinsics(frames)
        # print(extrinsics)
        
        # depth_image = cv2.flip(depth_image, 1)
        # color_image = cv2.flip(color_image, 1)
        # infrared_image = cv2.flip(infrared_image, 1)
        # print('Before calulate values')
        # bounding_box_points_color_image, length, width, height, point_cloud = calculate_values(calibration_output, depth_frame)
        # print('After calulate values and before visualise measurements')
        # visualise_measurements(color_image, bounding_box_points_color_image, length, width, height)
        # visualise_measurements_multiple(frames_devices, bounding_box_points_color_image, length, width, height)
        # print('After visualise measurements')
        
        # depth_image[depth_image == 0] = 256

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
        # np.savetxt(os.path.join(folder_path, 'depth_image.csv'), depth_image, delimiter=',', fmt='%d')
        # np.savetxt(os.path.join(folder_path, 'depth_colormap.csv'), depth_colormap, delimiter=',', fmt='%d')
        # break

        cv2.imshow('RGB image', color_image)
        # cv2.imshow('Depth image', depth_colormap)
        # cv2.imshow('Infrared image', infrared_image)

        if cv2.waitKey(1) == ord('q'):
            break

        if cv2.waitKey(1) == ord('m') and robot_moving:
            print(f'The robot has reached position {curr_img}...')
            move_robot = False
            robot_moving = False
        elif robot_moving:
            continue
        if curr_img <= n_imgs and not move_robot: #time_elapsed > 5 and curr_img <= n_imgs:
            store_images(curr_img, depth_colormap, depth_image, color_image,
                         rgb_folder_path, depth_folder_path, depth_values_folder_path)
            # calibration_output = ct.run_calibration(frame)
            # print(ct.run_calibration)
            print(f'Depth and RGB image for pos {curr_img} has been stored. \n')
            curr_img += 1
            move_robot = True
            timer = time.time()
        elif curr_img <= n_imgs and move_robot:
            # Move the robot here...
            print(f'Move the robot to position {curr_img}. \nPress M when it has reached the position...')
            robot_moving = True

        elif curr_img > n_imgs and not message_sent:
            print(f'{n_imgs} images has been stored.')
            message_sent = True

        if cv2.waitKey(1) == ord('c'):
            imgs_list = [file for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg'))]
            [os.remove(os.path.join(folder_path, file)) for file in imgs_list] 
            print(f'Deleted {len(imgs_list)} images.')
            curr_img = 0 
    cv2.destroyAllWindows()



def store_images(curr_img, depth_image, depth_values, color_image, 
                 rgb_folder_path, depth_folder_path, depth_values_folder_path):

    color_img_path = os.path.join(rgb_folder_path, f'color_img_pos_{curr_img}.png')
    depth_img_path = os.path.join(depth_folder_path, f'depth_img_pos_{curr_img}.png')
    depth_values_path = os.path.join(depth_values_folder_path, f'depth_values_pos_{curr_img}.npy')

    cv2.imwrite(color_img_path, color_image)
    cv2.imwrite(depth_img_path, depth_image)
    np.save(depth_values_path, depth_values)

    cv2.waitKey(500)

if __name__ == '__main__':
    # main_thread = threading.Thread(target=main)
    # main_thread.start()

    # other_thread = threading.Thread(target=run_calibration)
    # other_thread.start()

    main()