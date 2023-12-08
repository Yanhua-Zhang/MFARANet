# MFARANet
This repository includes the official project of our paper submitted to IEEE Transactions on Image Processing. Title: "Multi-Level Feature Aggregation and Recursive Alignment Network for Real-Time Semantic Segmentation".

## Usage

### 0. To be noted:

- We will gradually optimize the code to make it more readable and standardized.

- If you have any suggestions for improvement or encounter any issues while using this code, please feel free to contact me: zhangyanhua@mail.nwpu.edu.cn

### 1. Download pre-trained Resnet models  

Download the pre-trained Resnet models and put them into this folder: './model_pretrained'.

- resnet18-deep-stem:[link](https://drive.google.com/file/d/1q1VBV37acIte0GynoS054BWfwwdx1NiZ/view?usp=sharing)

- resnet50-deep-stem:[link](https://drive.google.com/file/d/1OktRGqZ15dIyB2YTySLfOVtprerHgbef/view?usp=sharing)

### 2. Prepare data 

- Download the [Cityscapes](https://www.cityscapes-dataset.com/) dataset and put it into this folder: './Dataset'.

- Your directory tree should be like this:

```bash
cityscapes
   ├── gtFine
   │   ├── test
   │   ├── train
   │   └── val
   └── leftImg8bit
       ├── test
       ├── train
       └── val
```

### 3. Environment 

We trained our model on one NVIDIA GeForce GTX 3090 with the CUDA 11.1 and CUDNN 8.0.

- Python 3.8.13.

- PyTorch 1.8.1. 

- Please refer to 'requirements.txt' for other dependencies.

### 4. Our trained models on the Cityscapes

````bash
cd MFARANet-main
````

#### Trained on the Train set and inference on the Valuation set.

- Download the trained model:[link](https://drive.google.com/file/d/1vGLHOW-_ref28PC0LXSyMuW-J6QJRPLB/view?usp=sharing). On the Valuation set, this trained model reaches 78.2% and 77.9% with input size 1024x2048 and 1024x1024, respectively. Its pruned version reaches 77.9% and 77.7%, respectively.

- Put 'MFARANet_resnet_18_deep_stem_cityscapes_2GPU_train_lr_005_batch_14_200_200.pth' into this folder: './save/model/our_model_train_val'. Run the following order:

````bash
python train_val_inference_general.py --yaml_Name='MFARANet_train_valuation_Basic_Config.yaml' --train_gpu 0 --NAME_model 'MFARANet_resnet_18_deep_stem' --Marker 'Branch_1_2_3_4_Paper_Val' --load_trained_model='./save/model/our_model_train_val/MFARANet_resnet_18_deep_stem_cityscapes_2GPU_train_lr_005_batch_14_200_200.pth'
````

#### Trained on the Train + Val set and inference on the Test set.

- Download the trained model:[link](https://drive.google.com/file/d/155ygZ50a6EwGqjm6qEnZ1shGQ9J3TKxn/view?usp=sharing). On the Test set, this trained model reaches 77.3% and 77.1% with input size 1024x2048 and 1024x1024, respectively. Its pruned version reaches 77.0% and 76.8%, respectively.

- Put 'MFARANet_resnet_18_deep_stem_cityscapes_2GPU_train_lr_005_batch_14_275_274.pth' into this folder: './save/model/our_model_trainval_test'. Run the following script:

````bash
python train_val_inference_general.py --yaml_Name='MFARANet_train_valuation_Basic_Config_Trainval_Test.yaml' --train_gpu 1 --NAME_model 'MFARANet_resnet_18_deep_stem' --Marker 'Branch_1_2_3_4_Paper_TrainVal_Test' --load_trained_model='./save/model/our_model_trainval_test/MFARANet_resnet_18_deep_stem_cityscapes_2GPU_train_lr_005_batch_14_275_274.pth'
````

- Run the following script to transform the predicted segmentation map from trainID to labelID, then follow the instruction in the [Cityscapes official website](https://www.cityscapes-dataset.com/) to submit to their test service:

````bash
CUDA_VISIBLE_DEVICES=0 python Submit_Cityscapes_trainID_2_labelID.py --path_Name='MFARANet_resnet_18_deep_stem_Branch_1_2_3_4_Paper_TrainVal_Test_cityscapes'
````

### 5. Train/Test by yourself

```bash
cd MFARANet-main
```

#### Train on the Train set and inference on the Valuation set.

````bash
python train_val_inference_general.py --yaml_Name='MFARANet_train_valuation_Basic_Config.yaml' --train_gpu 0 1 --NAME_model 'MFARANetScaleChoice_resnet_18_deep_stem' --Branch_Choose 1 2 3 4 --Marker 'Branch_1_2_3_4_Drop_0.05_Train_epochs_200' --epochs 200 --if_train_val
````

- We find that adding a small Dropout rate will prompt the performance.

````bash
python train_val_inference_general.py --yaml_Name='MFARANet_train_valuation_Basic_Config.yaml' --train_gpu 0 1 --NAME_model 'MFARANetScaleChoice_resnet_18_deep_stem' --Branch_Choose 1 2 3 4 --Dropout_Rate_CNN 0 0.05 0.05 0.05 0.05 --Marker 'Branch_1_2_3_4_Drop_0.05_Train_epochs_200' --epochs 200 --if_train_val
````

#### Train on the Train + Val set and inference on the Test set.

````bash
python train_val_inference_general.py --yaml_Name='MFARANet_train_valuation_Basic_Config_Trainval_Test.yaml' --train_gpu 0 1 --NAME_model 'MFARANetScaleChoice_resnet_18_deep_stem' --Branch_Choose 1 2 3 4 --Marker 'Branch_1_2_3_4_Drop_0.05_TrainVal_epochs_200' --epochs 200 --if_train_val
````

- Adding a small Dropout rate.

````bash
python train_val_inference_general.py --yaml_Name='MFARANet_train_valuation_Basic_Config_Trainval_Test.yaml' --train_gpu 0 1 --NAME_model 'MFARANetScaleChoice_resnet_18_deep_stem' --Branch_Choose 1 2 3 4 --Dropout_Rate_CNN 0 0.05 0.05 0.05 0.05 --Marker 'Branch_1_2_3_4_Drop_0.05_TrainVal_epochs_200' --epochs 200 --if_train_val
````

### 6. Measure FLOPs and FPS 

We adopt the code from [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR?v=2) to calculate FLOPs. When testing FPS, please make sure no other processes are hogging the GPU and CPU.

```bash
CUDA_VISIBLE_DEVICES=0 python model_summary_GFLOPs_params_FPS.py 
```

### 7. Links to the code used to reproduce the methods in Table XI

ENet: [unofficial](https://github.com/davidtvs/PyTorch-ENet), ESPNet: [official](https://github.com/sacmehta/ESPNet), ContextNet: [unofficial](https://github.com/zh320/realtime-semantic-segmentation-pytorch)

Fast-SCNN: [unofficial](https://github.com/Tramac/Fast-SCNN-pytorch), DFANetB: [unofficial](https://github.com/jandylin/DFANet_PyTorch), EDANet: [unofficial](https://github.com/wpf535236337/pytorch_EDANet)

CGNet M3N21: [unofficial](https://github.com/davidtvs/PyTorch-ENet), CIFReNet: [official](https://github.com/sacmehta/ESPNet), MiniNet-V2: [unofficial](https://github.com/zh320/realtime-semantic-segmentation-pytorch)

AGLNet: [unofficial](https://github.com/davidtvs/PyTorch-ENet), SGCPNet2: [official](https://github.com/sacmehta/ESPNet), ERFNet: [unofficial](https://github.com/zh320/realtime-semantic-segmentation-pytorch)

ICNet: [unofficial](https://github.com/davidtvs/PyTorch-ENet), SwiftNet-18: [official](https://github.com/sacmehta/ESPNet), SwiftNet-18pyr: [unofficial](https://github.com/zh320/realtime-semantic-segmentation-pytorch)

ShelfNet: [unofficial](https://github.com/davidtvs/PyTorch-ENet), SFNet(DF1): [official](https://github.com/lxtGH/SFSegNets), STDC2-Seg75: [unofficial](https://github.com/zh320/realtime-semantic-segmentation-pytorch)

DMA-Net: [unofficial](https://github.com/davidtvs/PyTorch-ENet), DeepLab-V2: [official](https://github.com/sacmehta/ESPNet), RefineNet: [unofficial](https://github.com/zh320/realtime-semantic-segmentation-pytorch)

PSPNet: [official](https://github.com/hszhao/semseg), BiSeNet-V1: [official](https://github.com/sacmehta/ESPNet), DFN: [unofficial](https://github.com/zh320/realtime-semantic-segmentation-pytorch)

HRNet-V2: [official](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR?v=2)

## Reference
* [PSPNet](https://github.com/hszhao/semseg)
* [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR?v=2)
* [SFNet](https://github.com/lxtGH/SFSegNets)

## Citations

```bibtex

xxx

```
