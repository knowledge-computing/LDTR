from process_shp import write_shp_in_imgcoord, read_shp
import os
import cv2

if __name__ == '__main__':
#     indrange_test = [10, 13, 32, 34, 8, 16] # bray wt test patches
#     indrange_test = [5, 7, 11] # louisville wt test patches
#     indrange_test = [27,6,16] # louisville rr test patches
    indrange_test = [43]
    
#     shp_dir = '/data/weiweidu/data/USGS_data/CO_Louisville_1965'
#     shp_name = 'Louisville_railroads_perfect_1965_4269.shp'
#     shp_name = 'Louisville_waterlines_perfect_1965.shp'
    shp_dir = '/data/weiweidu/data/USGS_data/CA_Bray_2001/ground_truth'
#     shp_name = 'CA_Bray_waterlines_2001_perfect.shp'
    shp_name = 'CA_Bray_railroads_2001_perfect_backup_4269.shp'
    
#     tif_dir = '/data/weiweidu/data/USGS_data/CO_Louisville_1965'
#     tif_name = 'CO_Louisville_450543_1965_24000_geo.tif'
    tif_dir =  '/data/weiweidu/data/USGS_data/CA_Bray_2001'
    tif_name = 'CA_Bray_100414_2001_24000_geo.tif' 
    
#     png_dir = '/data/weiweidu/data/USGS_data/CO_Louisville_1965'
#     png_name = 'CO_Louisville_1965_degeo.png'
    png_dir = tif_dir
    png_name = 'CA_Bray_100414_2001_24000_geo.png'
    
    out_dir = '/data/weiweidu/relationformer_connection_v4/data/ground_truth/usgs/railroads/shapefiles'
    out_name = 'ca_bray_2001_railroads_region'
    
    shp_path = os.path.join(shp_dir, shp_name)
    tif_path = os.path.join(tif_dir, tif_name)
    png_path = os.path.join(png_dir, png_name)
    
    polylines = read_shp(shp_path, tif_path)
    
    png_map = cv2.imread(png_path)
    height, width = png_map.shape[:2]
    count = 0
    
    row, col = 11550, 2048
    xmin, ymin = row, col
    xmax, ymax = row+2048, col+2048
    sub_polylines = []
    for line in polylines:
        sub_line = []
        for p in range(len(line)-1):
            if min(line[p][0], line[p+1][0]) < xmin or min(line[p][1], line[p+1][1]) < ymin or \
                max(line[p][0], line[p+1][0]) > xmax or max(line[p][1], line[p+1][1]) > ymax:
                continue
            x1, y1 = line[p][0] - row, line[p][1] - col
            x2, y2 = line[p+1][0] - row, line[p+1][1] - col
            if p == 0:
                sub_line.append([y1, x1])
                sub_line.append([y2, x2])
#                         sub_line.append(line[p])
#                         sub_line.append(line[p+1])
            else:
                sub_line.append([y2, x2])
#                         sub_line.append(line[p+1])
        if len(sub_line) > 1:
            sub_polylines.append(sub_line)

    out_path = os.path.join(out_dir, '%s%d.shp'%(out_name, indrange_test[count]))
#             print(sub_polylines)
    print('save in %s'%(out_path))
    print('-------------')
    write_shp_in_imgcoord(out_path, sub_polylines, tif_path)
    count += 1