import os
import cv2
import yaml
import json
from argparse import ArgumentParser
import pdb
import numpy as np
parser = ArgumentParser()
parser.add_argument('--config',
                    default=None,
                    help='config file (.yml) containing the hyper-parameters for training. '
                         'If None, use the nnU-Net config. See /config for examples.')
parser.add_argument('--checkpoint', default=None, help='checkpoint of the model to test.')
parser.add_argument('--device', default='cuda',
                        help='device to use for training')
parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[2],
                        help='list of index where skip conn will be made.')
parser.add_argument('--output_dir', type=str, help='Path to the directory save the prediction results.')

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

def ensure_format(bboxes):
    boxes_new = []
    for bbox in bboxes:
        if bbox[0] > bbox[2]:
            bbox[0], bbox[2] = bbox[2], bbox[0]
        if bbox[1] > bbox[3]:
            bbox[1], bbox[3] = bbox[3], bbox[1]
        
        # to take care of horizontal and vertical edges
        if bbox[2]-bbox[0]<0.2:
            bbox[0] = bbox[0]-0.075
            bbox[2] = bbox[2]+0.075
        if bbox[3]-bbox[1]<0.2:
            bbox[1] = bbox[1]-0.075
            bbox[3] = bbox[3]+0.075
            
        boxes_new.append(bbox)
    return np.array(boxes_new)

def test(args):
    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config['log']['message'])
    config = dict2obj(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))
    
    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    from monai.data import DataLoader
    from tqdm import tqdm
    import numpy as np

    from dataset_road_network import build_road_network_data
    from models import build_model
    from inference import relation_infer
    from utils import image_graph_collate_road_network

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda") if args.device=='cuda' else torch.device("cpu")

    net = build_model(config).to(device)

    test_ds, _ = build_road_network_data(
        config, mode='test'
    )

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
    with torch.no_grad():
        print('Started processing test set.')
        for batchdata in tqdm(test_loader):

            # extract data and put to device
            images, segs, nodes, edges = batchdata[0], batchdata[1], batchdata[2], batchdata[3]
            segs_np = segs.cpu().numpy()
            images_np = images.permute(0, 2, 3, 1).cpu().numpy()
            images_np = ((images_np * std + mean) * 255).astype('int32')
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
            batch_cnt = 0
            img_size = config.DATA.IMG_SIZE[0]
            for edges_, nodes_ in zip(pred_edges, pred_nodes):
                nodes_ = nodes_.cpu().numpy()
                pred_img = np.zeros((img_size, img_size, 3))
                for n in nodes_:
                    x, y = (n[0]*img_size).astype('int32'), (n[1]*img_size).astype('int32') 
                    cv2.circle(pred_img, (y, x), 3, (0,255,0), -1)
                for i_idx, j_idx in edges_:                 
                    n1, n2 = (nodes_[i_idx]*img_size).astype('int32'), (nodes_[j_idx]*img_size).astype('int32')
                    cv2.line(pred_img, (n1[1], n1[0]), (n2[1], n2[0]), (255,255,255), 1)
                if np.sum(pred_img) != 0:
                    patch_img = cv2.cvtColor(images_np[batch_cnt].astype('uint8'), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f'./{args.output_dir}/'+'%d.png' % cnt, pred_img)
                    cv2.imwrite(f'./{args.output_dir}/' +'%d_img.png' % cnt, patch_img)
                batch_cnt += 1
                cnt += 1

if __name__ == '__main__':
    args = parser.parse_args()
    test(args)