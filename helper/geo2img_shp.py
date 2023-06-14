from process_shp import read_shp, write_shp_in_imgcoord
import os

if __name__ == "__main__":
    for root, dirs, files in os.walk("/data/weiweidu/data/training_thesis", topdown=False):
        for fname in files:
            if "thrust_fault_line" in fname and '.shp' in fname:
                shp_path = os.path.join(root, fname)
                map_name = []
                temp = fname.split('_')
                for i in temp:
                    if i in ['fault', 'thrust', 'landslide']:
                        break
                    map_name.append(i)
                tif_path = os.path.join('/data/weiweidu/data/training/training', '_'.join(map_name)+'.tif')
#                 print(tif_path, " exists ", os.path.isfile(tif_path))
                polylines = read_shp(shp_path, tif_path)
                save_path = '/data/weiweidu/data/relationform/thrust_fault_lines/sat2graph_train/{}'.format(fname)
                write_shp_in_imgcoord(save_path, polylines, tif_path)
                