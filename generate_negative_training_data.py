import pdb
import math
import imageio
import pyvista
import numpy as np
import pickle
import random
import argparse
import os, cv2
import itertools
import glob
import networkx as nx
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000

patch_size = [256,256,1]
pad = [5,5,0]

def angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(np.clip(dot_product, a_min = -1, a_max=1))

def convert_graph(graph):
    node_list = []
    edge_list = []
    for n, v in graph.items():
        node_list.append(n)
    node_array = np.array(node_list)

    for ind, (n, v) in enumerate(graph.items()):
        for nei in v:
            idx = node_list.index(nei)
            edge_list.append(np.array((ind,idx)))
    edge_array = np.array(edge_list)
    return node_array, edge_array

vector_norm = 25.0 

def neighbor_transpos(n_in):
	n_out = {}

	for k, v in n_in.items():
		nk = (k[1], k[0])
		nv = []

		for _v in v :
			nv.append((_v[1],_v[0]))

		n_out[nk] = nv 

	return n_out 

def neighbor_to_integer(n_in):
	n_out = {}

	for k, v in n_in.items():
		nk = (int(k[0]), int(k[1]))
		
		if nk in n_out:
			nv = n_out[nk]
		else:
			nv = []

		for _v in v :
			new_n_k = (int(_v[0]),int(_v[1]))

			if new_n_k in nv:
				pass
			else:
				nv.append(new_n_k)

		n_out[nk] = nv 

	return n_out


def save_input(path, idx, patch, patch_seg, patch_coord, patch_edge):
    """[summary]

    Args:
        patch ([type]): [description]
        patch_coord ([type]): [description]
        patch_edge ([type]): [description]
    """
    cv2.imwrite(path+'/raw/sample_'+str(idx).zfill(6)+'_data.png', patch)
    cv2.imwrite(path+'/seg/sample_'+str(idx).zfill(6)+'_seg.png', patch_seg)
    
    if patch_coord != []:
        patch_edge = np.concatenate((np.int32(2*np.ones((patch_edge.shape[0],1))), patch_edge), 1)
        mesh = pyvista.PolyData(patch_coord)
        mesh.lines = patch_edge.flatten()
    else:
        mesh = pyvista.PolyData(np.transpose(np.array([[],[],[]])))
        mesh.lines = None #np.array([]) #
    # print(patch_edge.shape)
    mesh.save(path+'/vtp/sample_'+str(idx).zfill(6)+'_graph.vtp')


def patch_extract(save_path, image, mask, seg, mesh, stride):
    """[summary]

    Args:
        image ([type]): [description]
        coordinates ([type]): [description]
        lines ([type]): [description]
        patch_size (tuple, optional): [description]. Defaults to (64,64,64).
        num_patch (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    global image_id
    p_h, p_w, _ = patch_size
    pad_h, pad_w, _ = pad

    p_h = p_h -2*pad_h
    p_w = p_w -2*pad_w
    
    h, w, d = image.shape
    print('image height, width: ', h, w)
    h_n_interp, w_n_interp = h//stride, w//stride
    
    x_ = np.int32(np.linspace(5, h-5-p_h, h_n_interp)) 
    y_ = np.int32(np.linspace(5, w-5-p_w, w_n_interp)) 
    ind = np.meshgrid(x_, y_, indexing='ij')
    # Center Crop based on foreground
    for i, start in enumerate(list(np.array(ind).reshape(2,-1).T)):
#         start = np.array((start[0],start[1],0))
        start = np.array((start[1],start[0],0))
        end = start + np.array(patch_size)-1 -2*np.array(pad)
        
        patch = np.pad(image[start[0]:start[0]+p_h, start[1]:start[1]+p_w, :],\
                       ((pad_h,pad_h),(pad_w,pad_w),(0,0)), mode='constant')

        patch_list = [patch]
        
        patch_mask = np.pad(mask[start[0]:start[0]+p_h, start[1]:start[1]+p_w,],\
                            ((pad_h,pad_h),(pad_w,pad_w)), mode='constant')
        
        if np.sum(patch_mask) < 100:
            continue

        patch_seg = np.pad(seg[start[0]:start[0]+p_h, start[1]:start[1]+p_w,],\
                           ((pad_h,pad_h),(pad_w,pad_w)), mode='constant')

        seg_list = [patch_seg]

        # collect all the nodes
        bounds = [start[0], end[0], start[1], end[1], -0.5, 0.5]

        clipped_mesh = mesh.clip_box(bounds, invert=False)
        patch_coordinates = np.float32(np.asarray(clipped_mesh.points))
        patch_edge = clipped_mesh.cells[np.sum(clipped_mesh.celltypes==1)*2:].reshape(-1,3)

        patch_coord_ind = np.where((np.prod(patch_coordinates>=start, 1)*np.prod(patch_coordinates<=end, 1))>0.0)
        patch_coordinates = patch_coordinates[patch_coord_ind[0], :]  # all coordinates inside the patch
        patch_edge = [tuple(l) for l in patch_edge[:,1:] if l[0] in patch_coord_ind[0] and l[1] in patch_coord_ind[0]]
        
        temp = np.array(patch_edge).flatten()  # flatten all the indices of the edges which completely lie inside patch
        temp = [np.where(patch_coord_ind[0] == ind) for ind in temp]  # remap the edge indices according to the new order
        patch_edge = np.array(temp).reshape(-1,2)  # reshape the edge list into previous format
        
        if patch_coordinates.shape[0] < 2 or patch_edge.shape[0] < 1:
            if patch.shape[0] != patch_size[0] or patch.shape[1] != patch_size[0]:
                continue
            for patch, patch_seg in zip(patch_list, seg_list):
                save_input(save_path, image_id, patch, patch_seg, [], [])
                image_id = image_id+1
        else:
            continue
            
def prune_patch(patch_coord_list, patch_edge_list):
    """[summary]

    Args:
        patch_list ([type]): [description]
        patch_coord_list ([type]): [description]
        patch_edge_list ([type]): [description]

    Returns:
        [type]: [description]
    """
    mod_patch_coord_list = []
    mod_patch_edge_list = []

    for coord, edge in zip(patch_coord_list, patch_edge_list):

        # find largest graph segment in graph and in skeleton and see if they match
        dist_adj = np.zeros((coord.shape[0], coord.shape[0]))
        dist_adj[edge[:,0], edge[:,1]] = np.sum((coord[edge[:,0],:]-coord[edge[:,1],:])**2, 1)
        dist_adj[edge[:,1], edge[:,0]] = np.sum((coord[edge[:,0],:]-coord[edge[:,1],:])**2, 1)

        # straighten the graph by removing redundant nodes
        start = True
        node_mask = np.ones(coord.shape[0], dtype=np.bool_)
        while start:
            degree = (dist_adj > 0).sum(1)
            deg_2 = list(np.where(degree==2)[0])
            if len(deg_2) == 0:
                start = False
            for n, idx in enumerate(deg_2):
                deg_2_neighbor = np.where(dist_adj[idx,:]>0)[0]

                p1 = coord[idx,:]
                p2 = coord[deg_2_neighbor[0],:]
                p3 = coord[deg_2_neighbor[1],:]
                l1 = p2-p1
                l2 = p3-p1
                node_angle = angle(l1,l2)*180 / math.pi
                dist1, dist2 = np.linalg.norm(p1 - p2), np.linalg.norm(p1 - p3)
#                 if (dist1 < 0.04 or dist2 < 0.04 or dist1+dist2 < 0.1): #node_angle > 170 and
                if (dist1 < 0.03 or dist2 < 0.03 or dist1+dist2 < 0.06): #node_angle > 170 and
                    node_mask[idx]=False
                    dist_adj[deg_2_neighbor[0], deg_2_neighbor[1]] = np.sum((p2-p3)**2)
                    dist_adj[deg_2_neighbor[1], deg_2_neighbor[0]] = np.sum((p2-p3)**2)

                    dist_adj[idx, deg_2_neighbor[0]] = 0.0
                    dist_adj[deg_2_neighbor[0], idx] = 0.0
                    dist_adj[idx, deg_2_neighbor[1]] = 0.0
                    dist_adj[deg_2_neighbor[1], idx] = 0.0
                    break
                elif n==len(deg_2)-1:
                    start = False

        new_coord = coord[node_mask,:]
        new_dist_adj = dist_adj[np.ix_(node_mask, node_mask)]
        new_edge = np.array(np.where(np.triu(new_dist_adj)>0)).T

        mod_patch_coord_list.append(new_coord)
        mod_patch_edge_list.append(new_edge)

    return mod_patch_coord_list, mod_patch_edge_list

def conflate_nodes(patch_coord_list, patch_edge_list):
    refine_patch_coord_list, refine_patch_edge_list = [], []
    for coord, edge in zip(patch_coord_list, patch_edge_list):
        dist_adj = np.zeros((coord.shape[0], coord.shape[0]))
        dist_all = np.zeros((coord.shape[0], coord.shape[0]))
        all_com = np.array([np.array(j) for j in itertools.combinations([i for i in range(coord.shape[0])], 2)])

        dist_all[all_com[:,0], all_com[:,1]] = np.sum((coord[all_com[:,0],:]-coord[all_com[:,1],:])**2, 1)
        dist_all[all_com[:,1], all_com[:,0]] = np.sum((coord[all_com[:,0],:]-coord[all_com[:,1],:])**2, 1) 
        
        dist_adj[edge[:,0], edge[:,1]] = np.sum((coord[edge[:,0],:]-coord[edge[:,1],:])**2, 1)
        dist_adj[edge[:,1], edge[:,0]] = np.sum((coord[edge[:,0],:]-coord[edge[:,1],:])**2, 1)
        
        node_mask = np.ones(coord.shape[0], dtype=np.bool_)
        dup_node_pairs = np.array(np.where((dist_all<0.0008))).T
        dup_node_pairs.sort()

        if dup_node_pairs.shape[0] == 0: # no duplicate nodes
            refine_patch_coord_list.append(coord)
            refine_patch_edge_list.append(edge)
            continue
        
        dup_dict = {} # group duplicate nodes
        
        for i, j in dup_node_pairs:
            new_comp = True
            for k, comp in dup_dict.items():
                if i in comp or j in comp:
                    dup_dict[k].add(i)
                    dup_dict[k].add(j)
                    new_comp = False
                    break
            if new_comp:
                dup_dict[i] = set([i])
                dup_dict[i].add(j)

        G = nx.from_edgelist(edge)
        G_p = G
        for k, comp in dup_dict.items():
            comp_idx = list(comp)
            if len(comp_idx) == 1: continue
            for j in comp_idx:
                if j == k or j not in list(G_p.nodes()): continue
                G_p = nx.contracted_nodes(G_p, k, j)
        new_edge = np.array([list(i) for i in G_p.edges()])

        for k, comp in dup_dict.items():
            comp_idx = list(comp)
            for j in comp_idx:
                if j == k: continue
                node_mask[j] = False

        refine_patch_coord_list.append(coord)
        refine_patch_edge_list.append(new_edge)
        
    return refine_patch_coord_list, refine_patch_edge_list

indrange_test = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process geophysics training data")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input data directory")
    parser.add_argument("--obj_name", type=str, required=True, help="Object name (e.g., 'fault_line')")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--start_index", type=int, default=1, help="Starting index for output filenames (default: 1)")
    parser.add_argument("--stride", type=int, default=128, help="Stride defines the cropping interval at which images are cut horizontally and vertically on a map.")
    args = parser.parse_args()
    
    root_dir = args.input_dir
    obj_name = args.obj_name
    
    image_id = args.start_index
    train_path = args.output_dir
    if not os.path.isdir(train_path):
        os.makedirs(train_path)
        os.makedirs(train_path+'/seg')
        os.makedirs(train_path+'/vtp')
        os.makedirs(train_path+'/raw')

    print('Preparing Negative Train Data')

    raw_files = []
    mask_files = []   
    seg_files = []
    vtk_files = []

    for f in os.listdir(root_dir):
        map_name, fmt = f.split('.')
        if 'tif' == fmt:
            raw_files.append(os.path.join(root_dir, f)) 
            
            seg_name = f"{map_name}_{obj_name}.png"
            seg_path = f"{root_dir}/{seg_name}"
            if os.path.exists(seg_path):
                seg_files.append(seg_path)
            
            mask_path = f"{root_dir}/{map_name}_map_segmentation.png"
            if os.path.exists(mask_path):
                mask_files.append(mask_path)
            
            vtk_path = f"{root_dir}/{map_name}_{obj_name}.p"
            if  os.path.exists(vtk_path):
                vtk_files.append(vtk_path)
                          
    # Ensure all file lists are of the same length
    assert len(seg_files) == len(raw_files) == len(vtk_files), \
        f"File count mismatch: {len(seg_files)} segmentation, {len(raw_files)} raw, {len(vtk_files)} VTKs"

    print(f"===> #Files: #maps={len(raw_files)}, #raster_gt={len(seg_files)}, #pickle={len(vtk_files)}")   

    for ind in range(len(raw_files)):
        print(ind, raw_files[ind])
        try:
            sat_img = cv2.imread(raw_files[ind])
        except:
            sat_img = cv2.imread(raw_files[ind][:-4]+".jpg")
        if mask_files == []:
            mask_img = np.ones(sat_img.shape[:2])
        else:
            mask_img = cv2.imread(mask_files[ind], 0) / 255
        with open(vtk_files[ind], 'rb') as f:
            graph = pickle.load(f)
        
        node_array, edge_array = convert_graph(graph)
        if node_array.size == 0:
            node_array = np.array([[2050, 2050], [2070, 2070]]).astype('float32')
            edge_array = np.array([[0, 1]])
        gt_seg = cv2.imread(seg_files[ind],0)
        
        patch_coord = np.concatenate((node_array, np.int32(np.zeros((node_array.shape[0],1)))), 1)  
        patch_edge = np.concatenate((np.int32(2*np.ones((edge_array.shape[0],1))), edge_array), 1)
        patch_coord = patch_coord.astype('int32')
        mesh = pyvista.PolyData(patch_coord)
        mesh.lines = patch_edge.flatten()
        patch_extract(train_path, sat_img, mask_img, gt_seg, mesh, args.stride)

    