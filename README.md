# **LDTR: Linear Object Detection Transformer**  
### **Accurate Graph Generation by Learning N-hop Connectivity Information**

## **Goal**
LDTR aims to detect **linear objects** in **topographic** or **geological maps**.

---

## **Docker Image for Training & Testing LDTR**
### **Run LDTR Docker Image**
#### **1. Pull LDTR Docker Image from Docker Hub**
```bash
nvidia-docker pull weiweiduan/ldtr_pytorch:versions
```

#### **1. Run LDTR docker image** 
```bash
nvidia-docker run -t -i -v {local_dir}:{docker_dir} -p 8888:8888 weiweiduan/ldtr_pytorch:version0 /bin/bash
```

## Training Data Generation

<code> python generate_noprune_data.py </code>

## Train LDTR

<code> python train.py --config configs/usgs_railroads.yaml --cuda_visible_device 0 </code>

To update the parameters for data, model architecture, and training process, please update './configs/usgs_railroads.yaml'


## Use LDTR to detect desired linear objects

<code> python test_darpa_map_conflation_shapely.py --cuda_visible_device 0 --config ./configs/usgs_railroads.yaml --checkpoint ./trained_weights/runsrailroads_1965_g256_10/models/checkpoint_epoch=200.pt </code>

Please update './configs/usgs_railroads.yaml' for the path to the testing images 

## **Training and Testing Data**

### **Training Data**
Here is a link to an example **training dataset**:

- **Training Data**: [Download Here](https://drive.google.com/file/d/14JQzqCUrqmpNxITJwUhskyLM9yhbnY4z/view?usp=sharing)
  
#### **Setup Instructions**
1. **Download and unzip** the training data
2. **Update the configuration file** (`usgs_railroads.yaml`):
   - Locate the `DATA_PATH` field.
   - Set it to the **unzipped training data directory**.

---
### **Testing Data**
Here is a link to an example **testing dataset**:

- **Testing Data**: [Download Here](https://drive.google.com/file/d/1eaEMET5wi2i3arYEXuZWWlSgD52fo9Fd/view?usp=sharing)

#### **Contents of the Testing Dataset**
The zip file contains:
1. **`OR_Camas_g256_s100.zip`** → Direct input data for LDTR inference.
2. **`OR_Camas.tif`** → The original map used for testing.
3. **`OR_Camas_fault_line.tif`** → Ground truth raster image representing the **fault line** (linear object).

#### **Setup Instructions**
To test LDTR using this dataset:
1. **Unzip the testing data**
2. **Update the configuration file** (`usgs_railroads.yaml`):
   - Set `TEST_DATA_PATH` to the **unzipped testing data directory**.
   - Set `TEST_MAP_PATH` and `TEST_TIF_PATH` to the **original map path**.
