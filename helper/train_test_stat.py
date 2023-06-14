import os
import cv2
import numpy as np

if __name__ == "__main__":
#     root_dir = "/data/weiweidu/data/relationform/thrust_fault_lines/sat2graph_train"
#     root_dir = "/data/weiweidu/data/relationform/thrust_fault_lines/sat2graph_test"
#     root_dir = "/data/weiweidu/data/relationform/scarp_lines/sat2graph_train"
    root_dir = "/data/weiweidu/data/relationform/scarp_lines/sat2graph_test"
    
#     map_names = ['CO_Frisco','CA_BartlettSprings','ID_basement','CO_Bailey','AZ_PeachSprings',\
#     'CA_WhiteLedgePeak','AK_HowardPass','AK_Seldovia','VA_Lahore_bm','CA_NV_LasVegas',\
#     'AK_LookoutRidge','AK_Christian','AK_Ikpikpuk','CO_GreatSandDunes']
#     map_names = ['AK_HinesCreek','CO_Elkhorn','ID_LowerValley', 'CO_ClarkPeak']

#     map_names = ["CA_BartlettSprings", "CO_SanchezRes", "CA_BlackMtn", "CA_Sage", "CA_LosAngeles", "CO_Eagle",\
#         "CO_Granite", "CA_ProvidenceMtns"]
    map_names = ['CO_SanLuis', 'CO_BigCostilla']
    
    count = 0
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in map_names:
            for f in files:
                fmt = f.split('.')[1]
#                 if 'thrust_fault_line' in f and 'png' == fmt and name in f:
                if 'scarp_line' in f and 'png' == fmt and name in f:
                    seg = cv2.imread(os.path.join(root, f))
                    h, w = seg.shape[:2]
                    for r in range(0, h, 2048):
                        for c in range(0, w, 2048):
                            count += 1
    print('num of patches is {}'.format(count))
                           