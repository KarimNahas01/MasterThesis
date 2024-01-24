import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
from box_dimensioner_multicam.box_dimensioner_multicam_demo import run_calibration, calculate_values, visualise_measurements


def main():

    calibration_output = run_calibration()

    pipe = rs.pipeline()
    cfg = rs.config()

    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipe.start(cfg)

    n_imgs = 5
    curr_img = 1
    images_folder_name = 'img'
    connector = input('Input name of connector: ').replace(" ", "_")
    message_sent = False
    folder_path = os.path.join(images_folder_name, f'connector_{connector}')
    timer = time.time()
    move_robot = True

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

        frame = pipe.wait_for_frames()
        depth_frame = frame.get_depth_frame()
        color_frame = frame.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        depth_image = cv2.flip(depth_image, 1)
        # color_image = cv2.flip(color_image, 1)
        # print('Before calulate values')
        # bounding_box_points_color_image, length, width, height, point_cloud = calculate_values(calibration_output, color_frame)
        # print('After calulate values and before visualise measurements')
        # visualise_measurements(color_image, bounding_box_points_color_image, length, width, height)
        # print('After visualise measurements')
        
        # depth_image[depth_image == 0] = 256

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
        # np.savetxt(os.path.join(folder_path, 'depth_image.csv'), depth_image, delimiter=',', fmt='%d')
        # np.savetxt(os.path.join(folder_path, 'depth_colormap.csv'), depth_colormap, delimiter=',', fmt='%d')
        # break

        cv2.imshow('RGB image', color_image)
        cv2.imshow('Depth image', depth_colormap)

        if cv2.waitKey(1) == ord('q'):
            break
        if time_elapsed > 5 and curr_img <= n_imgs:
            store_images(curr_img, depth_colormap, depth_image, color_image,
                         rgb_folder_path, depth_folder_path, depth_values_folder_path)
            print(f'Depth and RGB image for pos {curr_img} has been stored')
            curr_img += 1
            move_robot = True
            timer = time.time()
        elif curr_img <= n_imgs and move_robot:
            # Move the robot here...
            print(f'Robot is moving to pos {curr_img}...')
            move_robot = False
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
    main()