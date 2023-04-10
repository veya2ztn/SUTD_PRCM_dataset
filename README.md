# SUTD_PRT_dataset

This is the SUTD PRT dataset for Metasurface Machine Learning.

The full dataset is a 8G file and can be download here:

<!--   -  [Google drive](https://drive.google.com/file/d/1mVUwwFr0atS0nHSRjyCFCXDW-627iUxI/view?usp=sharing)  -->
- [PLG DATASET] https://drive.google.com/file/d/1Q69avKkWlpHimfsPT-kQdHOL73PcQmiO/view?usp=sharing, 
- [PLR DATASET] https://drive.google.com/file/d/1WdQfb3yqLbusvXdtvonK4pGObyXSx7QO/view?usp=sharing, 
- [PTN DATASET] https://drive.google.com/file/d/1aunyk9f-oLIV5Zp4jp4a4SAnQB79U8Vp/view?usp=sharing, 
- [RDN DATASET] https://drive.google.com/file/d/1nZzlnJfyBN2ukchr-f-1RwsDKB8WKazB/view?usp=sharing
## Introduction

This dataset is consist of 30k+60k+60k+110k 「image($16\times16)$」- 「complex value curve $2\times1001$」pair.

<img src="https://github.com/veya2ztn/SUTD_PRT_dataset/blob/master/images/have_a_look_for_all_image_dataset.image.jpg?raw=true" alt="have_a_look_for_all_image_dataset.curve" style="zoom:15%;" /><img src="https://github.com/veya2ztn/SUTD_PRT_dataset/blob/master/images/have_a_look_for_all_image_dataset.curve.jpg?raw=true" alt="have_a_look_for_all_image_dataset.image" style="zoom: 20%;" />

Divide  along the pattern symmetry, we have four main image classes:

| Name | Volume |  Freedom  | Description                                                  |
| :--: | -----: | :-------: | :----------------------------------------------------------- |
| PLG  |  30000 | $2^{152}$ | The unit assembles as a polygon image, which must be a connected topo |
| PLR  |  60000 | $2^{???}$ | The unit assembles as a polygon image, which another polygon inside |
| PTN  |  60000 | $2^{102}$ | Combination of square, cross, triangle, U-shape, H-shape     |
| RDN  | 120000 | $2^{256}$ | All the units are randomly set 0 or 1                        |

![image-20210507014350020](https://github.com/veya2ztn/SUTD_PRT_dataset/blob/master/images/image-20210507014350020.png?raw=truehttps://github.com/veya2ztn/SUTD_PRT_dataset/blob/master/images/image-20210507014350020.png?raw=true)



## Usage

###  Completely use the dataset

1. download the dataset into `dataset_path` and unzip.

   The dataset structure tree is like:

   ```bash
   SUTDPRTDATASET
   ├── PLGDATASET
   │   ├── full_data_list
   │   │   ├── Data_001
   │   │   │   ├── Integrate_curve_1.npy
   │   │   │   └── Integrate_image_1.npy
   │   │   ├── Data_002
   │   │   │   ├── Integrate_curve_2.npy
   │   │   │   └── Integrate_image_2.npy
   │   ........
   │   ........
   ├── RDNDATASET
   │   ├── full_data_list
   │   │   ├── Data_001
   │   │   │   ├── Integrate_curve_1.npy
   │   │   │   └── Integrate_image_1.npy
   │   │   ├── Data_002
   │   │   │   ├── Integrate_curve_2.npy
   │   │   │   └── Integrate_image_2.npy
   │  ........
   │  ........
   ```

2. Use `ln -s real_data_path load_data_path` put the data what you want to use into a new file.

   For example, 

   ```bash
   PLGDATASET
   ├── full_data_list
   │     ........
   │     ........
   ├── train_data_list                            
   │   ├── Data_001 -> ../full_data_list/Data_001 
   │   └── Data_002 -> ../full_data_list/Data_002 
   └── valid_data_list                            
       ├── Data_001 -> ../full_data_list/Data_003 
       └── Data_002 -> ../full_data_list/Data_004 
   ```

3. use the `SMSDatasetN` or `SMSDatasetC` module load the dataset path. This module will automatedly load the data in the assigned path. For  example.

   ```python
   dataset_train = SMSDatasetN("data/PLGDATASET/train_data_list",None)
   dataset_train = SMSDatasetN("data/PLGDATASET/valid_data_list",None)
   ```

4. More option please see the `class SMSDataset` in `dataset_module.py`

### Fast use this dataset.

We provide fast train/test dataset script. 

- B1NE class: The fast task for 
  - Transmission curve norm: so the value now is real
  - Precision is 0.001

For example,

```python
from dataset import get_FAST_B1NE_dataset
dataset_path = "data"
dataset_class= "RDN" # can choose ["RDN","PTN","PLR","PLG","PLG250"]
curve_feature=  128  # assign the reduced dimenstion, fast mode only support [32,128,1001]
# for simple norm-curve  
dataset_train,dataset_valid = get_FAST_B1NE_dataset(dataset_path,
                                                    dataset=dataset_class,FeatureNum=curve_feature)

# for simple binary classifation problem: MPJ  
dataset_train,dataset_valid = get_FAST_B1NE_dataset(dataset_path,
                                         dataset=dataset_class,FeatureNum=curve_feature,                                                              type_predicted="onehot",target_predicted="balance_leftorright")

# for peak parameter prediction  
dataset_train,dataset_valid = get_FAST_B1NE_dataset(dataset_path,
                                         dataset=dataset_class,FeatureNum=curve_feature,                                                              type_predicted="combination",target_predicted="peakparameters")
```

If you use this dataset in your research, please use below citation:
```
@article{zhang2022symmetry,
  title={Symmetry Enhanced Network Architecture Search for Complex Metasurface Design},
  author={Zhang, Tianning and Kee, Chun Yun and Ang, Yee Sin and Li, Erping and Ang, Lay Kee},
  journal={IEEE Access},
  volume={10},
  pages={73533--73547},
  year={2022},
  publisher={IEEE}
}
```
