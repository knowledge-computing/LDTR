# **LDTR: Linear Object Detection Transformer**  
### **Accurate Graph Generation by Learning N-hop Connectivity Information**

## **Goal**
LDTR aims to detect **linear objects** in **topographic** or **geological maps**.

---

## **Docker Image for Training & Testing LDTR**
### **Run LDTR Docker Image**
#### **1. Pull LDTR Docker Image from Docker Hub**
```bash
nvidia-docker pull weiweiduan/ldtr_pytorch:version0
```

#### **1. Run LDTR docker image** 
```bash
nvidia-docker run -t -i -v {local_dir}:{docker_dir} -p 8888:8888 weiweiduan/ldtr_pytorch:version0 /bin/bash
```

## Training Data Generation

<!-- 
This command runs the `generate_positive_training_data.py` script with the following parameters:
- `--input_dir <path_to_input_dir>`: Specifies the directory containing the input data.
- `--obj_name <line_object_name>`: Defines the name of the object to be processed.
- `--output_dir <path_to_generated_training_data>`: Indicates the directory where the generated training data will be saved.
- `--start_index <starting_index_for_output_files>`: Sets the starting index for the output files.
- `--stride <stride_for_cropping_patches>`: Determines the stride value for cropping patches from the input data.
-->
### Generate positive data containing target lines
```bash
python generate_positive_training_data.py \
   --input_dir <path_to_input_dir> \
   --obj_name <line_object_name> \
   --output_dir <path_to_generated_training_data> \
   --start_index <starting_index_for_output_files> \
   --stride <stride_for_cropping_patches>
```

The above command runs the `generate_positive_training_data.py` script with the following parameters:
- `--input_dir <path_to_input_dir>`: Specifies the directory containing the input data.
- `--obj_name <line_object_name>`: Defines the name of the object to be processed.
- `--output_dir <path_to_generated_training_data>`: Indicates the directory where the generated training data will be saved.
- `--start_index <starting_index_for_output_files>`: Sets the starting index for the output files.
- `--stride <stride_for_cropping_patches>`: Determines the stride value for cropping patches from the input data.

### Generate negative data containing target lines
```bash
python generate_negative_training_data.py \
   --input_dir <path_to_input_dir> \
   --obj_name <line_object_name> \
   --output_dir <path_to_generated_training_data> \
   --start_index <starting_index_for_output_files> \
   --stride <stride_for_cropping_patches>
```

The above command runs the `generate_negative_training_data.py` script with the following parameters:
- `--input_dir <path_to_input_dir>`: Specifies the directory containing the input data.
- `--obj_name <line_object_name>`: Defines the name of the object to be processed.
- `--output_dir <path_to_generated_training_data>`: Indicates the directory where the generated training data will be saved.
- `--start_index <starting_index_for_output_files>`: Sets the starting index for the output files.
- `--stride <stride_for_cropping_patches>`: Determines the stride value for cropping patches from the input data.

**The expected input_dir should be**
```
/path_to_input_dir/
├── tif_map_name_1.tif
├── tif_map_name_1_line_name.png
└── tif_map_name_1_line_name.p
├── tif_map_name_2.tif
├── tif_map_name_2_line_name.png
└── tif_map_name_2_line_name.p
└── ...
```
- `<map_name>.tif` is the map image.
- `<map_name>_<line_name>.png` is the raster ground truth.
- `<map_name>_<line_name>.p` is the vector ground truth.

** The output_dir structure is
```
/path_to_generated_training_data/
├── raw/
│   ├── sample_000001_data.png
│   ├── sample_000002_data.png
│   └── ...
└── seg/
|  ├── sample_000000_seg.png
|  ├── sample_000001_seg.png
|  └── ...
└── vtp/
|  ├── sample_000000_graph.vtp
|  ├── sample_000001_graph.vtp
|  └── ...
```
- `raw/` contains map image patches.
- `seg/` contains raster linear object ground truth patches.
- `vtp/` contains vector linear object ground truth patches.

#### Here is a link to an example training dataset:

- **Training Data for thrust fault lines**: [Download Here](https://drive.google.com/file/d/1hw5q0r7xaB85P1ki1mHeJ4Ok0UKzn3-X/view?usp=sharing)
  
**Setup Instructions**
1. **Download and unzip** the training data
2. **Update the configuration file** 
   - Replace `./config/usgs_railroads.yaml` with the one for thrust fault line in the 
   `Model weights` table below
   - Locate the `DATA_PATH` field in the config file and set it to the **unzipped training data directory**.

## Train LDTR

<code> python train.py --config configs/usgs_railroads.yaml --cuda_visible_device 0 </code>

To update the parameters for data, model architecture, and training process, please update './configs/usgs_railroads.yaml'

## Model weights
| Object Name| Config file | Model weights | Model card |
|------------|---------|--------------|---------------|
| Railroads   | [Download](https://drive.google.com/drive/folders/1rE1S2xplSDw1RyOJwd5hewrvm-FHbC1d?usp=sharing)    | [Download](https://drive.google.com/file/d/1NlWojcF9zM_0YxVV9lVGDh6waCscCQMD/view?usp=sharing)   |  |
| Waterlines | [Download](https://drive.google.com/drive/folders/1rE1S2xplSDw1RyOJwd5hewrvm-FHbC1d?usp=sharing)    | [Download](https://drive.google.com/file/d/1Riy4X8VBhh5cx-1T-3YIpHXvW1sfwIH9/view?usp=sharing)   |  |
| Scarp lines | [Download](https://drive.google.com/drive/folders/1rE1S2xplSDw1RyOJwd5hewrvm-FHbC1d?usp=sharing)    | [Download](https://drive.google.com/file/d/1Yjb9CtZZPyOHBNbVUltVcU7p6RwOA7Hw/view?usp=sharing)  | |
| Thrust fault lines| [Download](https://drive.google.com/drive/folders/1rE1S2xplSDw1RyOJwd5hewrvm-FHbC1d?usp=sharing)    | [Download](https://drive.google.com/file/d/1GRp-qSYTG8Eh_eBAQxiQ4lvE2Q-pYU1A/view?usp=sharing)   | |

## Use LDTR to detect desired linear objects

### Generate Testing Data
Testing data is generated by cropping images from a map. The image file names should follow the format `<row>_<col>.png`, where `<row>` and `<col>` represent the row and column indices of the top-left corner of the cropped patches in the original map.

**The expected input_dir structure for testing dataset is**
```
/path_to_output_dir/
├── 0_0.png
├── 0_1.png
├── 1_0.png
└── 1_1.png
└── ...
```
- Each file represents a cropped patch from the input map.

#### Here is a link to an example testing dataset:

- **Testing Data**: [Download Here](https://drive.google.com/file/d/1O7ym9QCPMPtG-IIBqO4rJUkZjLdRgxuK/view?usp=sharing)
- The original input map and raster ground truth can be [downloaded here](https://www.dropbox.com/sh/0wh2mi87gmg2g93/AAAuRn3rPxs7KHZghgTVRv1oa?dl=0). The file names are `CO_Alamosa.png` and `CO_Alamosa_thrust_fault_line.png`.

**Setup Instructions**
to test LDTR using this dataset:
1. **Unzip the testing data**
2. **Update the configuration file** (`usgs_railroads.yaml`):
   - Set `TEST_DATA_PATH` to the **unzipped testing data directory**.
   - Set `TEST_MAP_PATH` to the **original map path**.


## Inference Using LDTR

```bash
python inference_on_map.py --cuda_visible_device 0 \
         --config <path_to_yaml_config_file> \
         --checkpoint <path_to_pretrained_LDTR> \
         --predict_raster \
         --predict_vector \
         --line_feature_name <target_line_name> \
         --prediction_dir <the_folder_to_save_the_detection> \
         --map_name <input_map_file_name>
```

The above command runs the `inference_on_map.py` script with the following parameters:
- `--cuda_visible_device 0`: Specifies the GPU device to be used.
- `--config <path_to_yaml_config_file>`: Path to the YAML configuration file.
- `--checkpoint <path_to_pretrained_LDTR>`: Path to the pretrained LDTR model checkpoint.
- `--predict_raster`: Flag to enable raster prediction.
- `--predict_vector`: Flag to enable vector prediction.
- `--map_name <input_map_file_name>`: Name of the input map file.
- `--line_feature_name <target_line_name>`: Name of the target line feature to detect. The detection output will be named as `<map_name>_<line_feature_name>`
- `--prediction_dir <the_folder_to_save_the_detection>`: Directory to save the detection results.

Please update './configs/usgs_railroads.yaml' for the path to the testing images 

## Inference Using LDTR on Map Patches
For debugging purposes, we provide a script to generate direct detection results at the patch level using LDTR.
```bash
python inference_on_patch.py --cuda_visible_device 0 \
         --config <path_to_yaml_config_file> \
         --checkpoint <path_to_pretrained_LDTR> \
         --output_dir <dir_to_save_detection_patches>
```

The above command runs the `inference_on_patch.py` script with the following parameters:
- `--cuda_visible_device 0`: Specifies the GPU device to be used.
- `--config <path_to_yaml_config_file>`: Path to the YAML configuration file.
- `--checkpoint <path_to_pretrained_LDTR>`: Path to the pretrained LDTR model checkpoint.
- `--output_dir <dir_to_save_detection_patches>`: Directory to save the detection results for each patch.

Please update './configs/usgs_railroads.yaml' for the path to the testing images 
