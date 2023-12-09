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

We trained our model on two NVIDIA GeForce GTX 3090 with the CUDA 11.1 and CUDNN 8.0.

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

- Download the trained model:[link](https://drive.google.com/file/d/155ygZ50a6EwGqjm6qEnZ1shGQ9J3TKxn/view?usp=sharing). On the Cityscapes Test service, this trained model reaches 77.3% ([Anonymous link](https://www.cityscapes-dataset.com/anonymous-results/?id=1534b251deaea61a1fb6fe80442bbdc7a5ae5c6c63726b34df1e50b721021c0f)) and 77.1% ([Anonymous link](https://www.cityscapes-dataset.com/anonymous-results/?id=17e3872b3deed5cb98526669c295b00b13dd2823fcf084f7bf1b07e4666a2466)) with input size 1024x2048 and 1024x1024, respectively. Its pruned version reaches 77.0% ([Anonymous link](https://www.cityscapes-dataset.com/anonymous-results/?id=4724321f30a2aa67adcd0d654f442cbb3442b8e1480192d15f95e6ad756548c6)) and 76.8% ([Anonymous link](https://www.cityscapes-dataset.com/anonymous-results/?id=c053f8e91cbdfd89f478b594c460b4017c6f7cbc72c36cd2e09d236d6a128587)), respectively.

- Put 'MFARANet_resnet_18_deep_stem_cityscapes_2GPU_train_lr_005_batch_14_275_274.pth' into this folder: './save/model/our_model_trainval_test'. Run the following script:

````bash
python train_val_inference_general.py --yaml_Name='MFARANet_train_valuation_Basic_Config_Trainval_Test.yaml' --train_gpu 1 --NAME_model 'MFARANet_resnet_18_deep_stem' --Marker 'Branch_1_2_3_4_Paper_TrainVal_Test' --load_trained_model='./save/model/our_model_trainval_test/MFARANet_resnet_18_deep_stem_cityscapes_2GPU_train_lr_005_batch_14_275_274.pth'
````

- Run the following script to transform the predicted segmentation map from trainID to labelID, then follow the instruction in the [Cityscapes official website](https://www.cityscapes-dataset.com/) to submit to their test service:

````bash
CUDA_VISIBLE_DEVICES=0 python Submit_Cityscapes_trainID_2_labelID.py --path_Name='MFARANet_resnet_18_deep_stem_Branch_1_2_3_4_Paper_TrainVal_Test_cityscapes'
````

#### Reproduce the MFARANet-Pruned results in Table IX.

- The simplest way is to uncomment line 329 in the file './model/model_MFARANet.py', and rerun the above scripts. We will provide rewrite code of MFARANet-Pruned later.  

### 5. Train/Test by yourself

- We train our model with 14 batch size and 1024x1024 crop size, which needs around 46G Training Memory on two GTX 3090 GPUs. You can halve the batch size to save the training memory, but please also change the learning rate accordingly. 

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

ENet: [unofficial](https://github.com/davidtvs/PyTorch-ENet), ESPNet: [official](https://github.com/sacmehta/ESPNet), ContextNet: [unofficial](https://github.com/zh320/realtime-semantic-segmentation-pytorch), Fast-SCNN: [unofficial](https://github.com/Tramac/Fast-SCNN-pytorch), DFANetB: [unofficial](https://github.com/jandylin/DFANet_PyTorch) 

EDANet: [unofficial](https://github.com/wpf535236337/pytorch_EDANet), CGNet M3N21: [official](https://github.com/wutianyiRosun/CGNet), CIFReNet: [official](https://github.com/WxTu/CIFReNet), MiniNet-V2: [unofficial](https://github.com/zh320/realtime-semantic-segmentation-pytorch), AGLNet: [unofficial](https://github.com/zh320/realtime-semantic-segmentation-pytorch)

SGCPNet2: [official](https://github.com/zhouyuan888888/sgcpnet), ERFNet: [official](https://github.com/Eromera/erfnet_pytorch), ICNet: [unofficial](https://github.com/liminn/ICNet-pytorch/tree/master), SwiftNet-18: [official](https://github.com/orsic/swiftnet), ShelfNet: [official](https://github.com/juntang-zhuang/ShelfNet)

SFNet(DF1): [official](https://github.com/lxtGH/SFSegNets), STDC2-Seg75: [official](https://github.com/MichaelFan01/STDC-Seg), DMA-Net: [unofficial](https://github.com/haritsahm/pytorch-DMANet), DeepLab-V2: [unofficial](https://github.com/isht7/pytorch-deeplab-resnet), RefineNet: [unofficial](https://github.com/DrSleep/refinenet-pytorch)

PSPNet: [official](https://github.com/hszhao/semseg), BiSeNet-V1: [unofficial](https://github.com/CoinCheung/BiSeNet), DFN: [unofficial](https://github.com/lingtengqiu/Deeperlab-pytorch/blob/master/dfn.py), HRNet-V2: [official](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR?v=2)

## Reference
* [PSPNet](https://github.com/hszhao/semseg)
* [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR?v=2)
* [SFNet](https://github.com/lxtGH/SFSegNets)

## Citations

```bibtex

xxx

```
