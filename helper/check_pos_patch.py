import pdb
import math
import imageio
import pyvista
import numpy as np
import pickle
import random
import os
import cv2

# patch_size = [128,128,1]
patch_size = [256,256,1]
pad = [5,5,0]


indrange_train = [i for i in range(42)] #42


if __name__ == "__main__":
    root_dir = "/data/weiweidu/sat2graph_inputs/CA_Bray_2001_waterlines"
#     root_dir = "/data/weiweidu/sat2graph_inputs/CO_Louisville_1965_waterlines"

    image_id = 1


    raw_files = []
    seg_files = []
    vtk_files = []

    for ind in indrange_train:
        raw_files.append(root_dir + "/ca_bray_2001_waterlines_region%d" % ind)
        seg_files.append(root_dir + "/ca_bray_2001_waterlines_gt_region%d.png" % ind)
        vtk_files.append(root_dir + "/ca_bray_2001_waterlines_region%d.p" % ind)
        
    for ind in range(len(raw_files)):
        try:
            sat_img = imageio.imread(raw_files[ind]+".png")
        except:
            sat_img = imageio.imread(raw_files[ind]+".jpg")
        
        with open(vtk_files[ind], 'rb') as f:
            graph = pickle.load(f)

        gt_seg = imageio.imread(seg_files[ind])[:,:,0]
        
        if np.sum(gt_seg) > 0:
            print(ind)

    

#     else:
#         raise Exception("Test folder is non-empty")


        

