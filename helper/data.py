import os
import numpy as np
import cv2

for root, dirs, files in os.walk('/data/weiweidu/relationformer_map/data/darpa/scarp_lines/train_data_g256_noprune/raw', topdown=False):
    for name in files:
        img_path = os.path.join(root, name)
        img = cv2.imread(img_path)
        if img.shape[0] != 256 or img.shape[1] !=256:
            print(name)