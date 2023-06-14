# LDTR: Linear Object Detection Transformer for Accurate Graph Generation by Learning the N-hop Connectivity Information

## LDTR's goal is to detect linear objects in the topographic or geological maps

## Docker imagery to train/testing LDTR
**Here is the command to run the docker imagery**

<code>sudo nvidia-docker run -t -i -v {local_dir}:{docker_dir} -p 8888:8888 pytorch/pytorch:1.2-cuda10.0-cudnn7-devel</code>

## Training Data Generation

<code> python generate_noprune_data.py </code>

## Train LDTR

<code> python train.py --config configs/usgs_railroads.yaml --cuda_visible_device 0 </code>

To update the parameters for data, model architecture, and training process, please update './configs/usgs_railroads.yaml'


## Use LDTR to detect desired linear objects

<code> python test_darpa_map_conflation_shapely.py --cuda_visible_device 0 --config ./configs/usgs_railroads.yaml --checkpoint ./trained_weights/runsrailroads_1965_g256_10/models/checkpoint_epoch=200.pt </code>

Please update './configs/usgs_railroads.yaml' for the path to the testing images 

