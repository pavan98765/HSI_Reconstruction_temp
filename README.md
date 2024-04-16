# HSI Reconstruction

This repository contains the implementation of a U-Net-based architecture augmented with spatial and spectral attention mechanisms for HyperSpectral Imaging (HSI) Reconstruction. The model aims to enhance the feature extraction process by focusing on relevant spatial and spectral components, improving the reconstruction quality of HSI data.

### Dataset links:

ARAD 1K dataset

training spectral images: [drive link](https://drive.google.com/file/d/1FQBfDd248dCKClR-BpX5V2drSbeyhKcq/view)

training RGB images:[drive link](https://drive.google.com/file/d/1A4GUXhVc5k5d_79gNvokEtVPG290qVkd/view)

validation spectral images:[drive link](https://drive.google.com/file/d/12QY8LHab3gzljZc3V6UyHgBee48wh9un/view)

validation RGB images:[drive link](https://drive.google.com/file/d/19vBR_8Il1qcaEZsK42aGfvg5lCuvLh1A/view)

testing RGB images:[drive link](https://drive.google.com/file/d/1A5309Gk7kNFI-ORyADueiPOCMQNTA7r5/view)

Please place the dataset in the below format.

```
|--dataset
    |--Train_Spec
        |--ARAD_1K_0001.mat
        |--ARAD_1K_0002.mat
        ：
        |--ARAD_1K_0950.mat
  	|--Train_RGB
        |--ARAD_1K_0001.jpg
        |--ARAD_1K_0002.jpg
        ：
        |--ARAD_1K_0950.jpg
    |--Test_RGB
        |--ARAD_1K_0951.jpg
        |--ARAD_1K_0952.jpg
        ：
        |--ARAD_1K_1000.jpg
    |--split_txt
        |--train_list.txt
        |--valid_list.txt
```

## Features

- **U-Net Architecture**: A modified U-Net architecture designed for HSI data processing.
- **Attention Mechanisms**: Incorporates attention modules to enhance model performance.
- **Output Normalization**: Includes an output normalization module to scale the output within a specific range.

## Requirements

To install the necessary requirements, you can use the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model, please place the dataset in the folder and run the training notebook

### Evaluation

The pretrained model is provided to evaluate it , please run the evaluation notebook

thanks
