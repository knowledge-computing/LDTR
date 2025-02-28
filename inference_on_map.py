import os
import cv2
import yaml
import json
from argparse import ArgumentParser
import pdb
import numpy as np
import collections
from shapely.ops import *
from shapely import geometry
from shapely.strtree import STRtree
from shapely.geometry import LineString, MultiLineString
from helper.process_shp import rm_dup_lines, conflate_lines, geometry_to_coordinates, remove_small_gaps
import geopandas
import torch
import geojson

parser = ArgumentParser()
parser.add_argument('--config',
                    default=None,
                    help='config file (.yml) containing the hyper-parameters for training. '
                         'If None, use the nnU-Net config. See /config for examples.')
# parser.add_argument('--trained_model_dir', default=None, help='directory storing trained models')
parser.add_argument('--checkpoint', default=None, help='checkpoint of the model to test.')
parser.add_argument('--device', default='cuda',
                        help='device to use for training')
parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[3],
                        help='list of index where skip conn will be made.')
parser.add_argument('--predict_raster', action='store_true', help='predict results are png if True')
parser.add_argument('--predict_vector', action='store_true', help='predict results are geojson if True')
parser.add_argument('--line_feature_name', default='fault_line', type=str, help='the name of line feature')
parser.add_argument('--prediction_dir', type=str, default='./pred_maps',
                       help='the path to save prediction results')
parser.add_argument('--map_name', type=str, default=None)
parser.add_argument('--buffer', type=int, default=10,
                        help='the buffer size for nodes conflation')


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

def add_lines(line2add, line_list, args):
    for i in range(len(line_list)):
        line = line_list[i]
        if line.distance(line2add) > 1:
            continue
        print('processing {}'.format(line2add))
        buffered_line = line.buffer(args.buffer)
        intersect = buffered_line.intersection(line2add)
        if not intersect:
            continue
        elif intersect.equals(line2add):
            print('duplicate')
            return line_list
        else:
            line_list[i] = line.union(line2add)
            print('union')
            return line_list
    line_list.append(line2add)
    return line_list

def add_lines_sindex(line2add, line_list, args):
    tree = STRtree(line_list)
    query_line = line2add.buffer(args.buffer)
    flag = False
    for o in tree.query(query_line):
        if o.intersects(query_line):
            intersect = o.intersection(query_line)
            o_n1, o_n2 = list(o.coords)
            o_n1, o_n2 = geometry.Point(o_n1), geometry.Point(o_n2)
            n12add, n22add = list(line2add.coords)
            n12add, n22add  = geometry.Point(n12add), geometry.Point(n22add)
            #highly overlapped/duplicated lines
            if o.buffer(args.buffer).contains(line2add): #(o_n1.distance(line2add) < 10 and o_n2.distance(line2add) < 10)
                return line_list  
            if query_line.contains(o):
                idx = line_list.index(o)
                del line_list[idx]
                flag = True
                break
                
    query_n1, query_n2 = list(line2add.coords)
    query_n1, query_n2 = geometry.Point(query_n1), geometry.Point(query_n2)
    buffered_query_n1, buffered_query_n2 = query_n1.buffer(args.buffer), query_n2.buffer(args.buffer)
    line4query_n1, line4query_n2 = None, None
    if flag:
        tree = STRtree(line_list)
    for o in tree.query(buffered_query_n1):
        if o.intersects(buffered_query_n1):
            line4query_n1 = o
            break
    for o in tree.query(buffered_query_n2):
        if o.intersects(buffered_query_n2):
            line4query_n2 = o
            break
    if line4query_n1 and line4query_n2:
        n11, n12 = list(line4query_n1.coords)
        n11, n12 = geometry.Point(n11), geometry.Point(n12)
        n21, n22 = list(line4query_n2.coords)
        n21, n22 = geometry.Point(n21), geometry.Point(n22)
        dist = np.array([geometry.LineString([n11, n21]).length, geometry.LineString([n11, n22]).length, \
                geometry.LineString([n12, n21]).length, geometry.LineString([n12, n22]).length])
        argmin = np.argmin(dist)
        if argmin == 0:
            line_list.append(geometry.LineString([n11, n21]))
        if argmin == 1:
            line_list.append(geometry.LineString([n11, n22]))    
        if argmin == 2:
            line_list.append(geometry.LineString([n12, n21]))    
        if argmin == 3:
            line_list.append(geometry.LineString([n12, n22]))  
        return line_list
    elif line4query_n1:
        n11, n12 = list(line4query_n1.coords)
        n11, n12 = geometry.Point(n11), geometry.Point(n12)
        dist = np.array([geometry.LineString([n11, query_n2]).length, geometry.LineString([n12, query_n2]).length])
        argmin = np.argmin(dist)
        if argmin == 0:
            line_list.append(geometry.LineString([n11, query_n2]))
        if argmin == 1:
            line_list.append(geometry.LineString([n12, query_n2]))
        return line_list
    elif line4query_n2:
        n21, n22 = list(line4query_n2.coords)
        n21, n22 = geometry.Point(n21), geometry.Point(n22)
        dist = np.array([geometry.LineString([n21, query_n1]).length, geometry.LineString([n22, query_n1]).length])
        argmin = np.argmin(dist)
        if argmin == 0:
            line_list.append(geometry.LineString([n21, query_n1]))
        if argmin == 1:
            line_list.append(geometry.LineString([n22, query_n1]))
        return line_list
        
    line_list.append(line2add)
    return line_list
            
    
def construct_graph(args, map_content_mask):    
    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config['log']['message'])
    config = dict2obj(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))

    config.DATA.PRED_MAP_NAME = args.map_name + f'_{args.line_feature_name}'
    
    import torch
    from monai.data import DataLoader
    from tqdm import tqdm
    import numpy as np

    from dataset_road_network import build_road_network_data
    from models import build_model
    from inference import relation_infer
    from metric_smd import StreetMoverDistance
    from metric_map import BBoxEvaluator
    from box_ops_2D import box_cxcywh_to_xyxy_np
    from utils import image_graph_collate_road_network

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda") if args.device=='cuda' else torch.device("cpu")
    
    net = build_model(config).to(device)

    test_ds, img_names = build_road_network_data(config, mode='test')

    test_loader = DataLoader(test_ds,
                            batch_size=config.DATA.TEST_BATCH_SIZE,
                            shuffle=False,
                            num_workers=config.DATA.NUM_WORKERS,
                            collate_fn=image_graph_collate_road_network,
                            pin_memory=True)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    net.eval()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    cnt = 0
    
    lines = [] # a list of LineString 
    
    with torch.no_grad():
        print('Started processing test set.')
        batch_cnt = 0
        for batchdata in tqdm(test_loader):
            # extract data and put to device
            images, segs, nodes, edges = batchdata[0], batchdata[1], batchdata[2], batchdata[3]
            segs_np = segs.cpu().numpy()
            images_np = images.permute(0, 2, 3, 1).cpu().numpy()
            images_np = ((images_np * std + mean) * 255).astype('int32')
#             print('gt shape', segs.shape)
            images = images.to(args.device,  non_blocking=False)
            segs = segs.to(args.device,  non_blocking=False)
            nodes = [node.to(args.device,  non_blocking=False) for node in nodes]
            edges = [edge.to(args.device,  non_blocking=False) for edge in edges]

            h, out, _, _, _, _ = net(images, seg=False)
            pred_nodes, pred_edges, _, pred_nodes_box, pred_nodes_box_score,\
            pred_nodes_box_class, pred_edges_box_score, pred_edges_box_class = \
            relation_infer(
                h.detach(), out, net, config.MODEL.DECODER.OBJ_TOKEN, config.MODEL.DECODER.RLN_TOKEN,
                nms=False, map_=True)
#             print('pred_edges shape: ', pred_edges)
            img_size = config.DATA.IMG_SIZE[0]
            for cnt, val in enumerate(zip(pred_edges, pred_nodes)):
                indices = img_names[batch_cnt*config.DATA.TEST_BATCH_SIZE+cnt][:-4].split('_')
                x_id, y_id = int(indices[-2]), int(indices[-1])
                patch_content_mask = map_content_mask[x_id:x_id+img_size, y_id:y_id+img_size]
                if np.sum(patch_content_mask) < 50: # filter out map legend areas
                    continue
                
                edges_, nodes_ = val
                nodes_ = nodes_.cpu().numpy()
                
                if edges_.shape[0] < 1:
                    continue
#                 print('positive  prediction')
                for i_idx, j_idx in edges_:                 
                    n1, n2 = (nodes_[i_idx]*img_size).astype('int32'), (nodes_[j_idx]*img_size).astype('int32')
                    indices = img_names[batch_cnt*config.DATA.TEST_BATCH_SIZE+cnt][:-4].split('_')
                    x_id, y_id = int(indices[-2]), int(indices[-1])

                    n1_in_map = [n1[0] + x_id, n1[1] + y_id]
                    n2_in_map = [n2[0] + x_id, n2[1] + y_id]
                    
                    line = geometry.LineString([[n1_in_map[1],n1_in_map[0]], [n2_in_map[1],n2_in_map[0]]])
                    lines = add_lines_sindex(line, lines, args)
            batch_cnt += 1
    return lines

def predict_png(args):
    """
    generate png prediction result
    """
    # Load the config files
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config)

    config.DATA.PRED_MAP_NAME = args.map_name + f'_{args.line_feature_name}_pred'
    
    map_name = args.map_name

    map_image = cv2.imread(config.DATA.TEST_MAP_PATH, 0)
    map_height, map_width = map_image.shape
    
    map_content_mask = np.ones((map_height, map_width)) 
    
    lines = construct_graph(args, map_content_mask)
 
    pred_png = np.zeros((map_height, map_width, 3))
    for line in lines:                       
        node1, node2 = list(line.coords)
        node1, node2 = [int(node1[0]), int(node1[1])], [int(node2[0]), int(node2[1])]
        cv2.line(pred_png, (node1[0], node1[1]), (node2[0], node2[1]), (255,255,255), 1)
    os.makedirs(args.prediction_dir, exist_ok=True)
    save_path = f'{args.prediction_dir}/{config.DATA.PRED_MAP_NAME}.png'
    cv2.imwrite(save_path, pred_png)
    print('*** save the predicted map in {} ***'.format(save_path))
    return True if np.sum(pred_png[:,:,0]/255) < 1000 else False

def predict_shp(args):
    """
    generate shp prediction
    """
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config)
                                        
    config.DATA.PRED_MAP_NAME = args.map_name + f'_{args.line_feature_name}'
    
    map_name = args.map_name
    
    map_image = cv2.imread(config.DATA.TEST_MAP_PATH, 0)
    map_height, map_width = map_image.shape
    
    map_content_mask = np.ones((map_height, map_width)) * 255
                                         
    lines = construct_graph(args, map_content_mask)
    nodup_lines = rm_dup_lines(lines)
    
    if not os.path.exists(args.prediction_dir):
        os.mkdir(args.prediction_dir)
    
    if len(nodup_lines) > 0:
        merged_lines = conflate_lines(nodup_lines)
        # merged_lines = group_lines_by_orientations(nodup_lines)
    else:
        merged_lines = conflate_lines(lines)
        # merged_lines = group_lines_by_orientations(nodup_lines)
    
    geojson_output_dir = f'{args.prediction_dir}/{args.map_name}'
    if not os.path.exists(geojson_output_dir):
        os.mkdir(geojson_output_dir)
    
    interm_geojson_path = f'{args.prediction_dir}/{config.DATA.PRED_MAP_NAME}.geojson'
    geojson_path = f'{args.prediction_dir}/{args.map_name}/{config.DATA.PRED_MAP_NAME}.geojson'

    refined_lines = remove_small_gaps(merged_lines)
    # refined_lines = merged_lines
    geometries = []
    properties = []
    for line in refined_lines:
        if isinstance(line, MultiLineString):
            temp_dict = {
                 "type": "MultiLineString",
                "coordinates": geometry_to_coordinates(line)
            }
        else:
            temp_dict = {
                "type": "LineString",
                "coordinates": geometry_to_coordinates(line)
            }
        geometries.append(temp_dict)
    # Create a GeoJSON feature collection
    feature_collection = geojson.FeatureCollection([
        geojson.Feature(geometry=geometry, properties=[]) for i, geometry in enumerate(geometries)
    ])

    with open(interm_geojson_path, "w") as f:
        geojson.dump(feature_collection, f)
    dash_pattern_dict = {}
    dash_pattern_dict['solid'] = refined_lines
    write_geojson_cdr(geojson_path, dash_pattern_dict, legend_text=description)
    print('*** save the predicted geojson in {} ***'.format(geojson_path))
    return geojson_path
    

if __name__ == '__main__':
    args = parser.parse_args()

    if args.predict_raster:
        predict_png(args)

    if args.predict_vector:
        output_shp_path = predict_shp(args)             

        