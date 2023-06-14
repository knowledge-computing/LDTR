import shutil
import os

if __name__ == "__main__":
#     src_dir = "/data/weiweidu/sat2graph_inputs/CO_Louisville_1965_waterlines"
#     src_dir = "/data/weiweidu/sat2graph_inputs/CO_Louisville_1965_railroads"
#     src_dir = "/data/weiweidu/sat2graph_inputs/CA_Bray_2001_waterlines"
#     tgt_dir = "/data/weiweidu/data/relationform/waterlines/sat2graph_train"
#     tgt_dir = "/data/weiweidu/data/relationform/railroads/sat2graph_test"
    src_dir = "/data/weiweidu/data/training_thesis"
    tgt_dir = "/data/weiweidu/data/relationform/thrust_fault_lines/sat2graph_train"
    
#     map_list = ["CA_BartlettSprings", "CO_SanchezRes", "CA_BlackMtn", "CA_Sage", "CA_LosAngeles", "CO_Eagle",\
#             "CO_Granite", "CA_ProvidenceMtns"] # for scarp lines
    map_list = ['CO_Frisco','CA_BartlettSprings','ID_basement','CO_Bailey','AZ_PeachSprings',\
                'CA_WhiteLedgePeak','AK_HowardPass','AK_Seldovia','VA_Lahore_bm','CA_NV_LasVegas',\
                'AK_LookoutRidge','CO_ClarkPeak','AK_Christian','AK_Ikpikpuk','CO_Elkhorn','CO_GreatSandDunes']
    
#     indrange_train = [i for i in range(42) if i not in [26,37,8,15,22,38]] # bray rr
#     indrange_train = [i for i in range(30) if i not in [27,6,16]] # louisville rr
#     indrange_train = [i for i in range(42) if i not in [10,13,32,34,8,16]] # bray wt
#     indrange_train = [i for i in range(25) if i not in [4,7,11]] # louisville wt

#     indrange_test = [26,37,8,15,22,38]
    
    if not os.path.isdir(tgt_dir):
        os.makedirs(tgt_dir)
    
    for root, dirs, files in os.walk(src_dir, topdown=False):
        for f_name in files:
            if f_name[:-4] in map_list:
#             if 'thrust_fault_line' in f_name and '.png' in f_name:
                src_raw_file = os.path.join(root, f_name)
                tgt_raw_file = os.path.join(tgt_dir, f_name)
                shutil.copy(src_raw_file, tgt_raw_file)
                            
#     for ind in indrange_train:
#         src_raw_file = src_dir + "/co_louisville_1965_waterlines_region%d.png" % ind
#         src_seg_file = src_dir + "/co_louisville_1965_waterlines_gt_region%d.png" % ind
#         src_vtk_file = src_dir + "/co_louisville_1965_waterlines_region%d.p" % ind
        
#         tgt_raw_file = tgt_dir + "/co_louisville_1965_waterlines_region%d.png" % ind
#         tgt_seg_file = tgt_dir + "/co_louisville_1965_waterlines_gt_region%d.png" % ind
#         tgt_vtk_file = tgt_dir + "/co_louisville_1965_waterlines_region%d.p" % ind

#         src_raw_file = src_dir + "/co_louisville_1965_railroads_region%d.png" % ind
#         src_seg_file = src_dir + "/co_louisville_1965_railroads_gt_region%d.png" % ind
#         src_vtk_file = src_dir + "/co_louisville_1965_railroads_region%d.p" % ind
        
#         tgt_raw_file = tgt_dir + "/co_louisville_1965_railroads_region%d.png" % ind
#         tgt_seg_file = tgt_dir + "/co_louisville_1965_railroads_gt_region%d.png" % ind
#         tgt_vtk_file = tgt_dir + "/co_louisville_1965_railroads_region%d.p" % ind

#         src_raw_file = src_dir + "/ca_bray_railroads_2001_region%d.png" % ind
#         src_seg_file = src_dir + "/ca_bray_railroads_2001_gt_region%d.png" % ind
#         src_vtk_file = src_dir + "/ca_bray_railroads_2001_region%d.p" % ind
        
#         tgt_raw_file = tgt_dir + "/ca_bray_2001_railroads_region%d.png" % ind
#         tgt_seg_file = tgt_dir + "/ca_bray_2001_railroads_gt_region%d.png" % ind
#         tgt_vtk_file = tgt_dir + "/ca_bray_2001_railroads_region%d.p" % ind
#         print("ca_bray_2001_waterlines_region%d.png" % ind)
#         shutil.copy(src_raw_file, tgt_raw_file)
#         shutil.copy(src_seg_file, tgt_seg_file)
#         shutil.copy(src_vtk_file, tgt_vtk_file)
        
#         print('copying {} to {}'.format(src_raw_file, tgt_raw_file))