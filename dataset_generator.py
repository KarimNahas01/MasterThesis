import numpy as np
import random
import json
import cv2
import os

CONSTANTS = json.load(open("constants.json"))

DIRECTORIES = CONSTANTS["directories"]

PARENT_FOLDER_NAME = DIRECTORIES["parent"]["folder_name"]

RGB_FOLDER_NAME = DIRECTORIES["rgb"]["folder_name"]
RGB_FILE_NAME = DIRECTORIES["rgb"]["file_name"]

ANNOTATIONS_FOLDER_NAME = DIRECTORIES["annotations"]["folder_name"]
ANNOTATIONS_FILE_NAME = DIRECTORIES["annotations"]["file_name"]

DATASET_STRUCTURE = CONSTANTS["dataset_structure"]

DATASET_PATH = f'{PARENT_FOLDER_NAME}/{DATASET_STRUCTURE["name"]}'

SUB_DIR_NAMES = DATASET_STRUCTURE['sub_dir_names']


def main():
    
    setup_dataset_directories()
    load_and_store_files()
    print('Dataset was generated.')

def setup_dataset_directories():
    n_imgs_train = len(os.listdir(f'{DATASET_PATH}/{DATASET_STRUCTURE["train"]["name"]}/{SUB_DIR_NAMES["images"]}'))

    for _, value in list(DATASET_STRUCTURE.items())[2:]:
        for dir in ["images", "labels"]:
            path = f'{DATASET_PATH}/{value["name"]}/{SUB_DIR_NAMES[dir]}'
            if n_imgs_train > 0:
                [os.remove(os.path.join(path, file)) for file in os.listdir(path)]
            else:
                os.makedirs(path, exist_ok=True) # This shouldnt be needed when ran through take_picture

def load_and_store_files(parent_folder=PARENT_FOLDER_NAME):
    file_map = os.listdir(parent_folder)
    filtered_folders = list(filter(lambda folder: folder.lower().endswith('connector'), file_map))

    for folder in filtered_folders:
        versions = os.listdir(f'{parent_folder}/{folder}')
        max_version = max(versions, key=lambda x: float(x.split('.')[1]))

        path = f'{parent_folder}/{folder}/{max_version}'

        rgb_files = sorted(os.listdir(f'{path}/{RGB_FOLDER_NAME}'), key=lambda x: int('_'.join(filter(str.isdigit, x))))
        txt_files = sorted(os.listdir(f'{path}/{ANNOTATIONS_FOLDER_NAME}'), key=lambda x: int('_'.join(filter(str.isdigit, x))))

        split_and_store(rgb_files, txt_files, path, folder)

def split_and_store(rgb_files, txt_files, path, folder):

    train_split = DATASET_STRUCTURE['train']['split']
    val_split = DATASET_STRUCTURE['val']['split']
    test_split = DATASET_STRUCTURE['test']['split']

    train_name = DATASET_STRUCTURE['train']['name']
    val_name = DATASET_STRUCTURE['val']['name']
    test_name = DATASET_STRUCTURE['test']['name']

    n_files = len(rgb_files)    

    num_train = int(train_split * n_files)
    num_val = int(val_split * n_files)
    num_test = n_files - num_train - num_val

    indices_list = np.arange(len(rgb_files))
    random.shuffle(indices_list)

    indices_dict = {}

    indices_dict.update({index: train_name for index in indices_list[:num_train]})
    indices_dict.update({index: val_name for index in indices_list[num_train:num_train + num_val]})
    indices_dict.update({index: test_name for index in indices_list[num_train + num_val:]})

    for idx, split in indices_dict.items():
        path_to_store = f'{DATASET_PATH}/{split}'

        img_path = f'{path}/{RGB_FOLDER_NAME}/{rgb_files[idx]}'
        txt_path = f'{path}/{ANNOTATIONS_FOLDER_NAME}/{txt_files[idx]}'

        img_name = img_path.split('/')[-1:][0]
        txt_name = txt_path.split('/')[-1:][0]

        img_file = cv2.imread(img_path)
        updated_file_name = (f'{img_name}').replace(RGB_FILE_NAME, folder)
        cv2.imwrite(f'{path_to_store}/{SUB_DIR_NAMES["images"]}/{updated_file_name}', img_file)
        
        updated_file_name = (f'{txt_name}').replace(ANNOTATIONS_FILE_NAME, folder)
        with open(txt_path, 'r') as file:
            txt_file = file.read()
        with open(f'{path_to_store}/{SUB_DIR_NAMES["labels"]}/{updated_file_name}', "w") as new_txt_file:
            new_txt_file.write(txt_file)

if __name__ == '__main__':
    main()