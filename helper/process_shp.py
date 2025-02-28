from osgeo import ogr, osr
import os
from osgeo import gdal
from osgeo.gdalconst import *
from shapely.ops import linemerge
from shapely.geometry import LineString, MultiLineString
from shapely.geometry import MultiPolygon, Polygon, Point
import math
from copy import deepcopy
import networkx as nx

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


def read_shp(shp_path, tif_path=None):
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
                        elif tif_path != None:
                            p = convert_to_image_coord(tmpt[0], tmpt[1], tif_path)
                        else:
#                             p = [-int(tmpt[1]), int(tmpt[0])]
                            p = [int(tmpt[0]), int(tmpt[1])]
                        polyline.append([int(p[0]), int(p[1])])
            elif points['type'] ==  "LineString":
                for i in points['coordinates']:
                    tmpt = i
                    if 'waterline' in shp_path:
                        p = convert_to_image_coord0(tmpt[0], tmpt[1], tif_path)
                    elif tif_path != None:
                            p = convert_to_image_coord(tmpt[0], tmpt[1], tif_path)
                    else:
#                         p = [-int(tmpt[1]), int(tmpt[0])]
                        p = [int(tmpt[0]), int(tmpt[1])]
                    polyline.append([int(p[0]), int(p[1])])

        count += 1
        polyline_list.append(polyline)
        f = layer.GetNextFeature()
    return polyline_list    

def write_shp_in_imgcoord(shp_name, lines):
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    
  # Getting shapefile driver
    driver = ogr.GetDriverByName('ESRI Shapefile')
    # Deleting the exit shapefile
    if os.path.exists(shp_name):
        driver.DeleteDataSource(shp_name)
    
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(4326)
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
    
def write_shp_in_imgcoord_output_schema(shp_name, lines):
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    
  # Getting shapefile driver
    driver = ogr.GetDriverByName('ESRI Shapefile')
    # Deleting the exit shapefile
    if os.path.exists(shp_name):
        driver.DeleteDataSource(shp_name)
    
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(4326)
    # Creating the shapefile
    ds = driver.CreateDataSource(shp_name)
    layer = ds.CreateLayer('layerName', spatial_reference, geom_type = ogr.wkbLineString)
    
    if ds is None:
        print ('Could not create file')
        sys.exit(1)
    
    fieldDefn = ogr.FieldDefn('ID', ogr.OFTInteger)
    layer.CreateField(fieldDefn)
    fieldDefn = ogr.FieldDefn('geometr', ogr.OFTString)
    layer.CreateField(fieldDefn)
    fieldDefn = ogr.FieldDefn('name', ogr.OFTString)
    layer.CreateField(fieldDefn)
    fieldDefn = ogr.FieldDefn('direction', ogr.OFTInteger)
    layer.CreateField(fieldDefn)
    fieldDefn = ogr.FieldDefn('type', ogr.OFTString)
    layer.CreateField(fieldDefn)
    fieldDefn = ogr.FieldDefn('descript', ogr.OFTString)
    layer.CreateField(fieldDefn)
    fieldDefn = ogr.FieldDefn('dash', ogr.OFTString)
    layer.CreateField(fieldDefn)
    fieldDefn = ogr.FieldDefn('symbol', ogr.OFTString)
    layer.CreateField(fieldDefn)
    
    
    cnt = 0
    for line in lines:
        cnt += 1
        lineString = ogr.Geometry(ogr.wkbLineString)
        if isinstance(line, list):
            for v in line:
                lineString.AddPoint(v[0], v[1])
        else:
            for p in list(line.coords) :
                lineString.AddPoint(p[0], p[1])

        featureDefn = layer.GetLayerDefn()
        feature = ogr.Feature(featureDefn)
        feature.SetGeometry(lineString)
        
        feature.SetField('ID', cnt)
        feature.SetField('geometr', 'line')
        feature.SetField('name', 'fault line')
        feature.SetField('direction', 0)
        feature.SetField('type', None)
        feature.SetField('descript', None)
        feature.SetField('dash', None)
        feature.SetField('symbol', None)
        
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
#                 elif line[i] == line[i-1]:
#                     print('same nodes')
#                 else:
#                     print('find a duplicate')
        else:
            if line[0] == line[1]:
#                 print('same nodes')
                continue
            if line not in no_dup_lines and [line[1], line[0]] not in no_dup_lines:
#                 print(line)
                no_dup_lines.append(line)
#             else:
#                 print('find a duplicate')
    return no_dup_lines

# def integrate_lines(lines):
#     merged_lines = [lines[0]]
#     for query_line in lines[1:]:
# #         query_nodes = list(query_line.coords)
# #         if query_nodes[0] == query_nodes[1]:
# #             print('same node')
# #             continue
# #         else:
#         for key_line in merged_lines:
#             temp_merged_line = linemerge([query_line, key_line])
#             if temp_merged_line.geom_type == 'MultiLineString':
#                 merged_lines.append(query_line)
#             else:
#                 merged_lines.append(temp_merged_line)
#     return merged_lines

# Define a function to check if two LineStrings share vertices
def is_overlapped(line1, line2):
    if isinstance(line1, list):
        line1 = LineString(line1)
        line2 = LineString(line2)
    return line1.intersects(line2)

def is_overlapped_within_buffer(line1, line2, buffer_distance):
    # Create buffer zones around line1 and line2
    buffer_line1 = line1.buffer(buffer_distance)
    buffer_line2 = line2.buffer(buffer_distance)
    if buffer_line1.intersects(buffer_line2):
        inters_geo = buffer_line1.intersection(buffer_line2)
        if inters_geo.geom_type == 'Point':
            return inters_geo.coords[0]
        elif inters_geo.geom_type == 'MultiPoint':
            return inters_geo[0].coords[0]
        elif inters_geo.geom_type == 'Polygon':
            return list(inters_geo.exterior.coords)[0]
        else:
            print(inters_geo.geom_type)
    else:
        return None
    

def integrate_lines(lines):
    merged_lines = LineString(lines[0])

    for _query_line in lines[1:]:
        query_line = LineString(_query_line)
        temp_merged = merged_lines.union(query_line)
        if temp_merged.geom_type != 'LineString':
            merged_lines = linemerge(temp_merged)
        else:
            merged_lines = temp_merged
    return list(merged_lines)


def conflate_lines(lines, refine_connect=True, connect_thres=4):
    refined_lines = []
    # Create a graph
    G = nx.Graph()
    # Add LineStrings as nodes to the graph
    for i, line in enumerate(lines):
#         line_shp = LineString(line)
        G.add_node(i, line=line)
    # Add edges between LineStrings that share vertices
    for i, line1 in enumerate(G.nodes(data='line')):
        for j, line2 in enumerate(G.nodes(data='line')):
            if i != j and is_overlapped(line1[1], line2[1]):
                G.add_edge(i, j)

    if refine_connect:# remove the edge which connected to more than connect_thres edges
        for node in list(G.nodes()):
            if G.degree(node) > connect_thres:
                G.remove_node(node)
    
    # Get connected components (groups of LineStrings with shared vertices)
    connected_components = list(nx.connected_components(G))
    # Iterate over each connected component
    for i, component in enumerate(connected_components):
        # Create a subgraph for the component
        subgraph = G.subgraph(component)
        # Remove vertices with degree > 2 within the component
        nodes_to_remove = [node for node in list(subgraph.nodes()) if subgraph.degree(node) > 2]
#         print(nodes_to_remove)
#         if nodes_to_remove != []:
#             print(G.nodes[nodes_to_remove[0]]['line'])
        refined_lines.extend([MultiLineString([G.nodes[ii]['line']]) for ii in nodes_to_remove])
        G.remove_nodes_from(nodes_to_remove)
    
    connected_components = list(nx.connected_components(G))
    for sub_component in connected_components:
        indices = list(sub_component)
        lines = [LineString(G.nodes[ii]['line']) for ii in indices]
        multi_line_string = MultiLineString(lines)
#         print(multi_line_string)
        refined_lines.append(multi_line_string)
#         polygon = multi_line_string.convex_hull
        
    return refined_lines

# Function to convert a MultiLineString or LineString to a list of coordinates
def geometry_to_coordinates(geometry):
    if isinstance(geometry, MultiLineString):
        coordinates = [list(line.coords) for line in geometry]
    elif isinstance(geometry, LineString):
        coordinates = list(geometry.coords)
    else:
        print(geometry.geom_type)
        raise ValueError("Input geometry must be a MultiLineString or LineString")
    return coordinates

def closest_point_to_multilinestring(point, multilinestring):
    point = Point(point)
    closest_distance = float('inf')  # Initialize closest distance to positive infinity
    closest_point = None  # Initialize closest point to None

    # Iterate through all LineString components of the MultiLineString
    for linestring in multilinestring.geoms:
        # Iterate through all points on the LineString
        for coords in linestring.coords:
            # Calculate the distance between the given point and the current point on the LineString
            distance = point.distance(Point(coords))
            # Check if the current point is closer than the previous closest point
            if distance < closest_distance:
                closest_distance = distance
                closest_point = Point(coords)

    return list(closest_point.coords)[0]

def merge_two_lines_with_buffered_points(line1, line2, intersect_point):
    close_pt1 = closest_point_to_multilinestring(intersect_point, line1)
    close_pt2 = closest_point_to_multilinestring(intersect_point, line2)
    line = MultiLineString([[close_pt1, close_pt2]])
#     all_lines = list(line1.geoms) + [line] + list(line2.geoms)
#     multi_lines = MultiLineString(all_lines)
    return line #multi_lines
    

def remove_small_gaps(lines, buffer_dist=20):
    refined_lines = []
    # Create a graph
    G = nx.Graph()
    # Add LineStrings as nodes to the graph
    for i, line in enumerate(lines):
#         line_shp = LineString(line)
        G.add_node(i, line=line)
    # Add edges between LineStrings that share vertices
    for i, line1 in enumerate(G.nodes(data='line')):
        for j, line2 in enumerate(G.nodes(data='line')):
            if i != j and is_overlapped(line1[1], line2[1]):
                G.add_edge(i, j)
    # Find nodes with degree < 2
    nodes_degree_less_than_2 = [node for node in G.nodes() if nx.degree(G, node) < 2]
    print(nodes_degree_less_than_2)
    for i in range(len(nodes_degree_less_than_2)):
        for j in range(i+1, len(nodes_degree_less_than_2)):
            ind1 = nodes_degree_less_than_2[i]
            ind2 = nodes_degree_less_than_2[j]
            line1 = G.nodes[ind1]['line']
            line2 = G.nodes[ind2]['line']
            intersected_point = is_overlapped_within_buffer(line1, line2, buffer_dist)
            if intersected_point:
                line_to_add = merge_two_lines_with_buffered_points(line1, line2, intersected_point)
                ind = len(G.nodes())
                G.add_node(ind, line=line_to_add)
                G.add_edge(ind1, ind)
                G.add_edge(ind, ind1)
                G.add_edge(ind2, ind)
                G.add_edge(ind, ind2)

    connected_components = list(nx.connected_components(G))
    for sub_component in connected_components:
        indices = list(sub_component)
        lines = []
        for ii in indices:
            lines.extend(list(G.node[ii]['line'].geoms))
        multi_line_string = MultiLineString(lines)
#         print(multi_line_string)
        refined_lines.append(multi_line_string)
    return refined_lines

def cal_orientations(line, bin_size=45):
    if isinstance(line, list):
        line = LineString(line)
    if isinstance(line, LineString):
        coords = list(line.coords)
    elif isinstance(line, MultiLineString):
        for line_string in line:
             coords = list(line_string.coords)
    # Calculate the difference between the start and end points
    p1 = [coords[-1][0], coords[-1][1]]
    p2 = [coords[0][0], coords[0][1]]
    temp_list = [p1[0], p2[0]]
    start_pt_ind = min(range(len(temp_list)), key=temp_list.__getitem__)
    if start_pt_ind == 0:
        dx = p1[1] - p2[1]
        dy = p1[0] - p2[0]
    else:
        dx = p2[1] - p1[1]
        dy = p2[0] - p1[0]
    # Calculate the angle using arctan
    angle = math.atan2(dy, dx)
    # Convert the angle from radians to degrees
    orientation = math.degrees(angle)
    # Ensure the orientation is between 0 and 360 degrees
    if orientation < 0:
        orientation += 360
    orientation_bin = orientation//bin_size
    return orientation_bin

def create_subgraphs_by_attribute(G, attribute="orientation"):
    # Get unique attribute values
    unique_values = set(nx.get_node_attributes(G, attribute).values())    
    # Create subgraphs for each unique attribute value
    subgraphs = []
    for value in unique_values:
        # Filter nodes based on attribute value
        nodes_with_attribute = [n for n, attrs in G.nodes(data=True) if attrs.get(attribute) == value]      
        # Create subgraph
        subgraph = G.subgraph(nodes_with_attribute)
        subgraphs.append(subgraph)    
    return subgraphs
    
def group_lines_by_orientations(lines, refine_connect=True, connect_thres=4):
    refined_lines = []
    # Create a graph
    G = nx.Graph()
    # Add LineStrings as nodes to the graph
    for i, line in enumerate(lines):
#         line_shp = LineString(line)
        degree = cal_orientations(line)
        G.add_node(i, line=line, orientation=degree)
    # Add edges between LineStrings that share vertices
    for i, line1 in enumerate(G.nodes(data='line')):
        for j, line2 in enumerate(G.nodes(data='line')):
            if i != j and is_overlapped(line1[1], line2[1]):
                G.add_edge(i, j)

    if refine_connect:# remove the edge which connected to more than connect_thres edges
        for node in list(G.nodes()):
            if G.degree(node) > connect_thres:
                G.remove_node(node)
    sub_graphs = create_subgraphs_by_attribute(G)
    for _sub_G in sub_graphs:
        sub_G = nx.Graph(_sub_G)
        # Get connected components (groups of LineStrings with shared vertices)
        connected_components = list(nx.connected_components(sub_G))
        # Iterate over each connected component
        for i, component in enumerate(connected_components):
            # Create a subgraph for the component
            subgraph = sub_G.subgraph(component)
            # get vertices with degree > 2 (i.e., intersections) within the component
            nodes_to_remove = [node for node in list(subgraph.nodes()) if subgraph.degree(node) > 2]
            refined_lines.extend([MultiLineString([sub_G.nodes[ii]['line']]) for ii in nodes_to_remove])
            sub_G.remove_nodes_from(nodes_to_remove)
        
        connected_components = list(nx.connected_components(sub_G))
        for sub_component in connected_components:
            indices = list(sub_component)
            lines = [LineString(sub_G.nodes[ii]['line']) for ii in indices]
            multi_line_string = MultiLineString(lines)
    #         print(multi_line_string)
            refined_lines.append(multi_line_string)
    return refined_lines