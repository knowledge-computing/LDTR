import os
import json
import mapdriver_usgs as md
import graph_ops_usgs as graphlib 
import pickle 

dataset_folder = '/data/weiweidu/data/relationform/thrust_fault_lines/sat2graph_train'

for root, dirs, files in os.walk(dataset_folder, topdown=False):
    for name in files:
        if 'osm' not in name:
            continue

        osm_path = os.path.join(root, name) 
        region_name = name[:-4]
        json_name = region_name+'.json'
        pickle_name = region_name+'.p'
        
        OSMMap = md.OSMLoader(False, osm_path, includeServiceRoad = False)

        node_neighbor = {} # continuous

#         print(OSMMap.nodedict)

        node_neighbor = {} # continuous

        for node_id, node_info in OSMMap.nodedict.items():
            lat = node_info["lat"]
            lon = node_info["lon"]
            n1key = (lat,lon)
#             print((int(lat),int(lon)))
#             cv2.circle(test, (int(lat),int(lon)), 5, 255, -1)
            neighbors = []
            for nid in list(node_info["to"].keys()) + list(node_info["from"].keys()) :
                if nid not in neighbors:
                    neighbors.append(nid)
#             print(neighbors)
            for nid in neighbors:
                n2key = (OSMMap.nodedict[nid]["lat"],OSMMap.nodedict[nid]["lon"])

                node_neighbor = graphlib.graphInsert(node_neighbor, n1key, n2key)


        # 			#graphlib.graphVis2048(node_neighbor,[lat_st,lon_st,lat_ed,lon_ed], "raw.png")

        # 			# interpolate the graph (20 meters interval)
        node_neighbor = graphlib.graphDensify(node_neighbor, density = 20)
#         print(node_neighbor)
        node_neighbor_region = graphlib.graph2RegionCoordinate(node_neighbor)
        # #         print(node_neighbor)
        # #         prop_graph = dataset_folder+"/region_%d_graph_gt.pickle" % c
        # #         pickle.dump(node_neighbor_region, open(prop_graph, "w"))

        #         #graphlib.graphVis2048(node_neighbor,[lat_st,lon_st,lat_ed,lon_ed], "dense.png")
        # #         graphlib.graphVis2048Segmentation(node_neighbor, [lat_st,lon_st,lat_ed,lon_ed], dataset_folder+"/region_%d_" % c + "gt.png")

        node_neighbor_refine, sample_points = graphlib.graphGroundTruthPreProcess(node_neighbor_region)
#         for loc, n_locs in node_neighbor_refine.items():
#             for n_loc in n_locs:
#                 print(n_loc)
#                 cv2.circle(test, (int(n_loc[0]),int(n_loc[1])), 5, 255, -1)
            #         print(node_neighbor_refine)
        with open(os.path.join(dataset_folder,pickle_name), "wb") as handle:
            pickle.dump(node_neighbor_refine, handle, 0)
        print(os.path.join(dataset_folder,json_name))
                
        json.dump(sample_points, open(os.path.join(dataset_folder,json_name), "w"), indent=2)
#         if 'region0' in name:
#             cv2.imwrite('test.jpg', test)
#             break