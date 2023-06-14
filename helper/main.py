from process_shp import read_shp_in_imgcoord, rm_dup_lines, write_shp_in_imgcoord
import geopandas

if __name__ == '__main__':
    tif_path = '/data/weiweidu/data/USGS_data/CO_Louisville_1965/CO_Louisville_450543_1965_24000_geo.tif'
#     in_shp_path = './pred4eval/shapefiles/ca_bray_2001_waterlines_pred_region34.shp'
#     in_shp_path = './data/ground_truth/usgs/waterlines/shapefiles/ca_bray_2001_waterlines_region34.shp'
    in_shp_path = '/data/weiweidu/CoANet_copy/run/darpa/CoANet-resnet/pred4eval/shapefiles/ID_LowerValley_thrust_fault_lines_pred.shp'
    out_shp_path = in_shp_path[:-4]+'_nodup.shp'
#     out_geojson_path = './pred4eval/shapefiles/ca_bray_2001_waterlines/ca_bray_2001_waterlines_region34.geojson'
    out_geojson_path = '/data/weiweidu/CoANet_copy/run/darpa/CoANet-resnet/pred4eval/shapefiles/thrust_fault_lines/ID_LowerValley_thrust_fault_lines.geojson'

    lines = read_shp_in_imgcoord(in_shp_path, tif_path)
    refined_lines = rm_dup_lines(lines)
    write_shp_in_imgcoord(out_shp_path, refined_lines, tif_path)
    print('--- save the shapefile in {}'.format(out_shp_path))
    shp_file = geopandas.read_file(out_shp_path)
    
    shp_file.to_file(out_geojson_path, driver='GeoJSON')
    print('--- save the geojson in {}'.format(out_geojson_path))