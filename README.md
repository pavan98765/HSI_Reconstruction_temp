# HSI Reconstruction

This repository contains the implementation of a U-Net-based architecture augmented with spatial and spectral attention mechanisms for HyperSpectral Imaging (HSI) Reconstruction. The model aims to enhance the feature extraction process by focusing on relevant spatial and spectral components, improving the reconstruction quality of HSI data.

## Dataset

The ARAD 1K dataset used for training, validation, and testing consists of RGB and spectral images. You can download the dataset from the following links:

- **Training Spectral Images**: [Google Drive](https://drive.google.com/file/d/1FQBfDd248dCKClR-BpX5V2drSbeyhKcq/view)
- **Training RGB Images**: [Google Drive](https://drive.google.com/file/d/1A4GUXhVc5k5d_79gNvokEtVPG290qVkd/view)
- **Validation Spectral Images**: [Google Drive](https://drive.google.com/file/d/12QY8LHab3gzljZc3V6UyHgBee48wh9un/view)
- **Validation RGB Images**: [Google Drive](https://drive.google.com/file/d/19vBR_8Il1qcaEZsK42aGfvg5lCuvLh1A/view)

- **Test RGB Images**: [Google Drive](https://drive.google.com/file/d/1A5309Gk7kNFI-ORyADueiPOCMQNTA7r5/view)

Please organize the downloaded dataset in the following format:

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

## Model Architecture

The model architecture is specifically designed for HyperSpectral Imaging (HSI) tasks, offering enhanced performance and feature extraction capabilities. Here's a breakdown of its components:

### Encoding

- **Spatial and Spectral Attention**: Each encoder block incorporates two attention mechanisms to focus on relevant spatial and spectral features, improving model performance.
- **Downsampling**: Convolutional layers followed by max-pooling operations reduce spatial dimensions while increasing feature depth.

### Bottleneck

- **Feature Refinement**: Two convolutional layers further refine encoded features, capturing high-level representations.

### Decoding

- **Upsampling**: Transposed convolutional layers recover spatial details lost during encoding, restoring image resolution.
- **Attention Mechanisms**: Attention modules applied to each decoder block refine reconstructed HSI, improving overall quality.
- **Output Normalization**: A normalization module ensures consistent scaling of the output, optimizing the reconstructed HSI data.

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
