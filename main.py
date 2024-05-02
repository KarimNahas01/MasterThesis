import yolo_model_generator as ymg
import dataset_generator as dg
import take_picture_new as tp
import annotation_tool as at
import dataset_tester as dt

import tkinter as tk
import cv2

from tkinter import messagebox

root = tk.Tk()
root.withdraw()

N_IMGS = 10

    # Priority (in order):
    # TODO (G): Add augmentation and preprocessing, and make the dataset account for it.
    # TODO (G): Add more augmentation steps.
        # Augmentation:
        #   - Apply blur
        #   - Vary brightness
        #   - Add noise
        # Preprocessing:
        #   - Normalizing
        #   - Checking size of image
    # TODO (K): Make the background interchangeable using transparent images. To reperesent different environments. (Take pictures from car for example)
    # TODO (K): Check the performance of using depth to extract the connector and the rod, and using colors to remove the white rod from the image. And then annotate that.
    # TODO: Clean up the code, make the dependencies less and correct naming on variables and methods etc.
    # TODO: Clean up the files and its structures
    # TODO: Add the constants to a generic python file so that it can be imported. Also add commonly used methods there.
    # TODO: Fix end call in robot program, so that it terminates automatically without the use of n_imgs

def main():
    collect_more = True
    while collect_more:
        tp.main(N_IMGS)
        cv2.destroyAllWindows()
        collect_more = messagebox.askyesno("Collect more", "Do you want to collect another connector?")

    print('Dataset collection completed')

    print('Initiated annotation')
    at.main()
    print('Annotation completed')
    cv2.destroyAllWindows()

    print('Initiated dataset generation')
    dg.main()
    print('Dataset generation completed')
    cv2.destroyAllWindows()

    print('Initiated training on the dataset')
    ymg.main()
    print('Training completed')
    cv2.destroyAllWindows()
    
    print('Initiated testing the dataset')
    dt.main()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()