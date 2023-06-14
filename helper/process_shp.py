from osgeo import ogr, osr
import os
from osgeo import gdal
from osgeo.gdalconst import *
import math
from copy import deepcopy

def convert_to_image_coord0(x, y, path): # convert geocoord to image coordinate
    dataset = gdal.Open(path, GA_ReadOnly)
    adfGeoTransform = dataset.GetGeoTransform()

    dfGeoX=float(x)
    dfGeoY =float(y)
    det = adfGeoTransform[1] * adfGeoTransform[5] - adfGeoTransform[2] *adfGeoTransform[4]

    X = ((dfGeoX - adfGeoTransform[0]) * adfGeoTransform[5] - (dfGeoY -
    adfGeoTransform[3]) * adfGeoTransform[2]) / det

    Y = ((dfGeoY - adfGeoTransform[3]) * adfGeoTransform[1] - (dfGeoX -
    adfGeoTransform[0]) * adfGeoTransform[4]) / det
    return [int(Y),int(X)]

def convert_to_image_coord(x, y, path): # convert geocoord to image coordinate
    ds = gdal.Open(path, GA_ReadOnly )
    target = osr.SpatialReference(wkt=ds.GetProjection())

    source = osr.SpatialReference()
    source.ImportFromEPSG(4269)

    transform = osr.CoordinateTransformation(source, target)

    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(x, y)
    point.Transform(transform)

    x, y = convert_to_image_coord0(point.GetX(), point.GetY(), path)
    return [x, y]


def read_shp(shp_path, tif_path):
    print('shp path: ', shp_path)
    print('tif path: ', tif_path)
    ds = ogr.Open(shp_path)
    layer = ds.GetLayer(0)
    f = layer.GetNextFeature()
    polyline_list = []
    count = 0
    while f:
        geom = f.GetGeometryRef()
        if geom != None:
        # points = geom.GetPoints()
            points = geom.ExportToJson()
            points = eval(points)
            polyline = []
            if points['type'] == "MultiLineString":
                for i in points["coordinates"]:
                    for j in i:
                        tmpt = j
                        if 'waterline' in shp_path:
                            p = convert_to_image_coord0(tmpt[0], tmpt[1], tif_path)
                        else:
                            p = convert_to_image_coord(tmpt[0], tmpt[1], tif_path)
                        polyline.append([int(p[0]), int(p[1])])
            elif points['type'] ==  "LineString":
                for i in points['coordinates']:
                    tmpt = i
                    if 'waterline' in shp_path:
                        p = convert_to_image_coord0(tmpt[0], tmpt[1], tif_path)
                    else:
                        p = convert_to_image_coord(tmpt[0], tmpt[1], tif_path)
                    polyline.append([int(p[0]), int(p[1])])

        count += 1
        polyline_list.append(polyline)
        f = layer.GetNextFeature()
    return polyline_list    

def write_shp_in_imgcoord(shp_name, lines, coor_path):
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    
  # Getting shapefile driver
    driver = ogr.GetDriverByName('ESRI Shapefile')
    # Deleting the exit shapefile
    if os.path.exists(shp_name):
        driver.DeleteDataSource(shp_name)
    
    srs = gdal.Open(coor_path).GetProjection()
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromProj4(srs)
    # Creating the shapefile
    ds = driver.CreateDataSource(shp_name)
    layer = ds.CreateLayer('layerName', spatial_reference, geom_type = ogr.wkbLineString)
    
    if ds is None:
        print ('Could not create file')
        sys.exit(1)
    
    fieldDefn = ogr.FieldDefn('fieldName', ogr.OFTReal)
    layer.CreateField(fieldDefn)
    
    cnt = 0
    for line in lines:
        cnt += 1
        lineString = ogr.Geometry(ogr.wkbLineString)
        if isinstance(line, list):
            for v in line:
                lineString.AddPoint(v[0], v[1])
        else:
            n1, n2 = list(line.coords)  
            lineString.AddPoint(n1[0], n1[1])
            lineString.AddPoint(n2[0], n2[1])
#         x, y = float(p[0]), float(p[1])
#         lineString.AddPoint(x,y)
        featureDefn = layer.GetLayerDefn()
        feature = ogr.Feature(featureDefn)
        feature.SetGeometry(lineString)
        feature.SetField('fieldName', 'LineString')
        layer.CreateFeature(feature)
        lineString.Destroy()
        feature.Destroy()
    ds.Destroy()
    print ("Shapefile created")    
    

def interpolation(start, end, inter_dis):
    dis = math.sqrt((start[0]-end[0])**2+(start[1]-end[1])**2)
    segment = []
    if dis == 0:
        return None
    elif dis <= inter_dis:
        return [start, end]
    else:
        ##### calculate k & b in y=kx+b
        add_num = round(dis/inter_dis, 0)   
        segment.append(start)
        if abs(end[1]-start[1]) < 5: ##### vertical line
            y_interval = int(round((end[0]-start[0])/float(add_num)))
            for i in range(1, int(add_num)):
                segment.append([start[0]+i*y_interval, start[1]])
        elif abs(end[0]-start[0]) < 5: ##### horizontal line
            x_interval = int(round((end[1]-start[1])/float(add_num)))
            for i in range(1, int(add_num)):
                segment.append([start[0], start[1]+i*x_interval])
        else:
            k = (end[1]-start[1]) / float(end[0]-start[0])
            b = end[1] - k*end[0]
#             x_interval = int(round((end[0]-start[0])/float(add_num)))
            x_interval = (end[0]-start[0])/float(add_num)
            for i in range(1, int(add_num)):
                new_x = start[0]+i*x_interval
                segment.append([int(new_x), int(k*new_x+b)])
        segment.append(end)

        return segment

def interpolate_polylines(polylines, inter_dis=16):
    polylines_interp = []
    for i, line in enumerate(polylines):
        line_interp = []
        for p in range(len(line)-1):
            x_s, y_s = line[p]
            x_e, y_e = line[p+1]
            vec_interp = interpolation([x_s, y_s], [x_e, y_e], inter_dis)
            if vec_interp == None:
                continue
            line_interp.extend(vec_interp)
        polylines_interp.append(line_interp)
    return polylines_interp
    
def construct_graph_on_map(polylines_interp): 
    # return nodes_dict: {1: [x, y], ...}
    # return edges_list: [(1,2),...]
    nodes_dict = {}
    edges_list = []
    counter = 0

    for i, line in enumerate(polylines_interp):
        for p in range(len(line)-1):
            x_s, y_s = line[p]
            x_e, y_e = line[p+1]
            if [x_s, y_s] not in nodes_dict.values():
                nodes_dict[counter] = [x_s, y_s]
                s_id = counter
                counter += 1
            else:
                s_id = list(nodes_dict.keys())[list(nodes_dict.values()).index([x_s, y_s])]
            if [x_e, y_e] not in nodes_dict.values():
                nodes_dict[counter] = [x_e, y_e]
                e_id = counter
                counter += 1
            else:
                e_id = list(nodes_dict.keys())[list(nodes_dict.values()).index([x_e, y_e])]
            edges_list.append((s_id, e_id))
    return nodes_dict, edges_list

def rm_dup_lines(lines):
    no_dup_lines = []
    for _line in lines:
        if isinstance(_line, list):
            line = _line
        elif _line.geom_type == 'LineString': # _line is shapely LineString
            line = list(_line.coords)
        else:
            line = [list(x.coords) for x in list(_line)]
            
        if len(line) > 2: # multi-lines
            for i in range(1, len(line)):
                if [line[i], line[i-1]] not in no_dup_lines and [line[i-1], line[i]] not in no_dup_lines:
                    no_dup_lines.append([line[i-1], line[i]])
                else:
                    print('find a duplicate')
        else:
            if line[0] == line[1]:
                print('same nodes')
                continue
            if line not in no_dup_lines and [line[1], line[0]] not in no_dup_lines:
#                 print(line)
                no_dup_lines.append(line)
            else:
                print('find a duplicate')
    return no_dup_lines
    