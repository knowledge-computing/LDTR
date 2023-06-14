import os
import cv2
import copy
import numpy as np

"""
[550, 300] bray railroads parallel for patch-scale
[12100, 2345] bray railroads parallel for map-scale
[550, 450] bray waterlines region32 for patch-scale
[10800, 4540] bray waterlines for map-scale
[900, 1330] louisville waterlines region7 for patch-scale
[2950, 5430] louisville waterlines for map-scale
[4180, 3250] CO_SanLuis scarp lines
[2550, 640] ID_LowerValley thrust fault lines for tps
[260, 335] louisville railroads fps for patch-scale
[10500, 4430] louisville railroads fps for map-scale
[1045, 3580] CO_ClarkPeak thrust fault lines FPs
[310,1536] louisville waterlines region11 for patch-scale for N-hop vis
[325, 0] louisville waterlines region11 for patch-scale for #tokens vis
"""

def buffer(pred_ske, buffer_size=1):
    if np.max(pred_ske)==0:
        print('blank img')
        return pred_ske, pred_ske
    if np.max(pred_ske) != 1:
        pred_ske= pred_ske/np.max(pred_ske)

    if buffer_size == 0:
        return pred_ske
    pred_copy = copy.deepcopy(pred_ske)
    for i in range(buffer_size):
        nonzeros_idx = np.where(pred_ske != 0)
        xs, ys = nonzeros_idx
        # west
        d1 = (xs-1, ys)
        # east
        d2 = (xs+1, ys)
        # north
        d3 = (xs, ys-1)
        # south
        d4 = (xs, ys+1)
        # northwest
        d5 = (xs-1, ys-1)
        # northeast
        d6 = (xs+1, ys-1)
        # southwest
        d7 = (xs-1, ys+1)
        # southeast
        d8 = (xs+1, ys+1)
        pred_ske[d1] = 1
        pred_ske[d2] = 1
        pred_ske[d3] = 1
        pred_ske[d4] = 1
        pred_ske[d5] = 1
        pred_ske[d6] = 1
        pred_ske[d7] = 1
        pred_ske[d8] = 1
    return pred_copy, pred_ske

if __name__ == '__main__':
    buffer_flag = False
    buffer_size = 2
    model_name = '4token'
#     input_dir = '/data/weiweidu/relationformer_connection_v4/pred4eval'
    input_dir = '/data/weiweidu/sat2graph_inputs/CO_Louisville_1965_waterlines'
#     input_dir = '/data/weiweidu/CoANet_copy/run/darpa/CoANet-resnet/pred4eval'
#     input_dir = '/data/weiweidu/SIINet/data/Seg/roadtracer_org/pred4eval'
#     input_dir = '/data/weiweidu/SIINet/data/Seg/roadtracer_org/prediction'
#     input_dir = '/data/weiweidu/relationformer_map_copy/pred4eval'
#     input_dir = '/data/weiweidu/relationformer_connection_darpa/pred4eval'
#     input_dir = '/data/weiweidu/data/training_png_shp'
#     input_dir = '/data/weiweidu/relationformer_connection_v4/pred4ablation/token80'

#     input_name = 'ca_bray_2001_railroads_pred_region43.png'
#     input_name = 'ca_bray_2001_railroads_gt_region43.png' #'ca_bray_2001_railroads_region43.png'
#     input_name = 'co_louisville_1965_waterlines_pred_region11.png'
    input_name = 'co_louisville_1965_waterlines_region11.png'
#     input_name = 'co_louisville_1965_waterlines_pred_region11.png'
#     input_name = 'co_louisville_1965_railroads_pred.png'
#     input_name = 'CA_Bray_railroads_2001_align_ijgis_siinet_probmap.jpeg'
#     input_name = 'CA_Bray_2001_waterlines_gtlabel_siinet_deconv_smallkernel_probmap_collar.png'
#     input_name = 'CO_Louisville_1965_waterlines_gtlabel_siinet_deconv_smallkernel_probmap_collar.png'
#     input_name = 'CO_Louisville_1965_railroads_gtlabel_siinet_deconv_smallkernel_probmap_e700.png'
#     input_name = 'ca_bray_2001_railroads_pred_region43.png'
#     input_name = 'co_louisville_1965_waterlines_pred_region7.png'
#     input_name = 'co_louisville_1965_waterlines_gt_region7.png'
#     input_name = 'CO_SanLuis_scarp_lines_pred.png'
#     input_name = 'ID_LowerValley_thrust_fault_line.png'
#     input_name = 'CO_ClarkPeak_thrust_fault_lines_pred.png'
    
    output_dir = '/data/weiweidu/relationformer_connection_v4/pred4vis'
    crop_x, crop_y = [325, 0]
    crop_size = 512
    input_path = os.path.join(input_dir, input_name)
    input_img = cv2.imread(input_path)
    if buffer_flag:
        input_img = input_img[:,:,0]
        input_img = np.pad(input_img, ((buffer_size+2, buffer_size+2), (buffer_size+2, buffer_size+2)))
        _, input_img = buffer(input_img, buffer_size)   
#         input_img = cv2.resize(input_img, (2048, 2048))
        input_img = input_img*255
        
    cropped_img = input_img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size]
#     cropped_img[cropped_img > 60] = 255
#     cropped_img[cropped_img <= 60] = 0
    if not model_name:
        cv2.imwrite(os.path.join(output_dir, input_name[:-4]+'_crop.png'), cropped_img)
    else:
        cv2.imwrite(os.path.join(output_dir, input_name[:-4]+'_crop_{}.png'.format(model_name)), cropped_img)
    


