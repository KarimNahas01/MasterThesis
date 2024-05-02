import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os
import torch
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
from torchvision.models.detection import fasterrcnn_resnet50_fpn
# from roboflow import Roboflow
from ultralytics import YOLO

USING_REALSENSE_CAMERA = True

RESOLUTION = [640, 480]
FRAMERATE = {'color':30, 'depth':30, 'infrared':30}

MODEL_PATH = 'runs/detect/train6/weights/best.pt'
# MODEL_PATH = 'C:/Users/karim/Downloads/best (2).pt'

FRAME_PATH = 'tmp/frame_read.png'

if USING_REALSENSE_CAMERA:
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.bgr8, FRAMERATE['color'])
    # config.enable_stream(rs.stream.depth, RESOLUTION[0], RESOLUTION[1], rs.format.z16, FRAMERATE['depth'])
    # config.enable_stream(rs.stream.infrared, RESOLUTION[0], RESOLUTION[1], rs.format.y8, FRAMERATE['infrared'])

    align_to = rs.stream.color # ----
    align = rs.align(align_to) # ----

    profile = pipeline.start(config)
    device = profile.get_device()

def main():
    # predict('img/white_connector/rgb_images/color_img_pos_1.png')
    # predict('EWASS_demo_img/pictures_for_demo_connector/v.1.0/rgb_images')
    # predict('EWASS_demo_img/pictures_for_demo_connector/v.2.0/rgb_images')
    if USING_REALSENSE_CAMERA:
        predict()
    else:  
        predict('testing_everything/dakr_gray_large_connector/v.1.0/rgb_images')
        # predict(f'EWASS_demo_img/pictures_for_demo_connector/v.1.0/rgb_images',
        #         save=True, folder_name='some_predictions', delay=1000)
        # predict('EWASS_demo_img/more_images_connector/v.1.0/rgb_images',
                # save=True, folder_name='some_predictions2', delay=1000)
    # while True:
    #     get_image()
    #     predict(FRAME_PATH)

def predict(path=None, save=False, folder_name=None, exist_ok=True, delay=0):

    model = YOLO(MODEL_PATH)

    exist_ok = not (not folder_name)

    if not path:
        # model.predict(source="0", imgsz=(736, 1088), show=True, conf=0.5)
        model.predict(source="0", save=save, name=folder_name, exist_ok=exist_ok, show=True, conf=0.5)

    elif os.path.isdir(path):
        file_map = enumerate(sorted(os.listdir(path), key=lambda x: int(''.join(filter(str.isdigit, x)))))
        for idx, filename in file_map:
            if any(filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
                img_path = os.path.join(path, filename)
                prediction_result = model.predict(img_path, name=folder_name, exist_ok=exist_ok, save=save, conf=0.5)
                cv2.imshow("Prediction Result", prediction_result[0].plot())
                cv2.waitKey(delay)
    else:
        prediction_result = model.predict(path, save=save, name=folder_name, exist_ok=exist_ok, conf=0.5)
        cv2.imshow("Prediction Result", prediction_result[0].plot())
        cv2.waitKey(1)

def predict_cv2():
    model = YOLO(MODEL_PATH)
    camera = cv2.VideoCapture(0)
    img_counter = 0

    while True:
        ret, frame = camera.read()

        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_path = "path/opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_path, frame)
            outs = model.predict(img_path)
            img_counter += 1

    camera.release()

'''
def get_image():
    global FRAME_PATH

    os.makedirs('tmp', exist_ok=True)

    frame = pipeline.wait_for_frames(timeout_ms=5000)
    aligned_frames = align.process(frame)

    frames = {
        # 'depth': aligned_frames.get_depth_frame(),
        'color': aligned_frames.get_color_frame()
    }

    # depth_image = np.asanyarray(frames['depth'].get_data())
    color_image = np.asanyarray(frames['color'].get_data())

    # depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)

    # return depth_image, color_image

    cv2.imwrite(FRAME_PATH, color_image)



def main_roboflow():
    rf = Roboflow(api_key="OXsBAWTljmd1VJvnr9g0")
    project = rf.workspace().project("connector-detector")
    model = project.version(1).model

    loop_start_time = time.time()

    while True:

        depth_image, color_image = get_image()
        
        os.makedirs('tmp', exist_ok=True)
        cv2.imwrite(f'tmp/frame_read.png', color_image)

        model.predict('tmp/frame_read.png', confidence=40, overlap=30).save("tmp/prediction.png.jpg")

        # model.predict("img/random_images_connector/v.1.0/rgb_images/color_img_1.png", confidence=40, overlap=30).save("prediction.jpg")

        loop_end_time = time.time()
        iteration_time = loop_end_time - loop_start_time

        cv2.imshow(f'RGB image', cv2.imread('tmp/prediction.png.jpg'))
        cv2.setWindowTitle('RGB image', f'RGB image ({round(1/iteration_time, 2)} FPS)')
        cv2.waitKey(1)
        
        loop_start_time = loop_end_time

def main_torch():

    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=91) 
    model.eval() 

    state_dict = torch.load('best.pt')

    # Filter out keys that don't match
    filtered_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

    # Update the model's state_dict with the loaded weights
    model.load_state_dict(filtered_dict, strict=False)

    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),  
    ])
    
    os.makedirs('tmp', exist_ok=True)
    depth_image, color_image = get_image()
    cv2.imwrite('tmp/frame_read.png', color_image)

    input_image = Image.open('tmp/frame_read.png')
    input_tensor = transform(input_image)

    with torch.no_grad():
        predictions = model([input_tensor])

    boxes = predictions[0]['boxes'].tolist()
    labels = predictions[0]['labels'].tolist()

    print(boxes)
    print(labels)

    
    input_image_np = np.array(input_image)

    for box, label in zip(boxes, labels):
        box = [int(coord) for coord in box]

        cv2.rectangle(input_image_np, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

        cv2.putText(input_image_np, str(label), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Bounding Boxes', input_image_np)
    cv2.waitKey(0)


    # draw = ImageDraw.Draw(input_image)
    # for box, label in zip(boxes, labels):
    #     draw.rectangle(box, outline="red")
    #     draw.text((box[0], box[1]), str(label), fill="red")
    # del draw

'''

if __name__ == '__main__':
    main()