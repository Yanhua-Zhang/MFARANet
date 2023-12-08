# MFARANet
This repository includes the official project of our paper submitted to IEEE Transactions on Image Processing. Title: "Multi-Level Feature Aggregation and Recursive Alignment Network for Real-Time Semantic Segmentation".

## Usage

### 0. To be noted:

- We will gradually optimize the code to make it more readable and standardized.

- If you have any suggestions for improvement or encounter any issues while using this code, please feel free to contact me: zhangyanhua@mail.nwpu.edu.cn

### 1. Download pre-trained Resnet models

Download the pre-trained Resnet models and put them into this folder: './model_pretrained'.

- resnet50-deep-stem:[link](https://drive.google.com/file/d/1OktRGqZ15dIyB2YTySLfOVtprerHgbef/view?usp=sharing)

- resnet18-deep-stem:[link](https://drive.google.com/file/d/1q1VBV37acIte0GynoS054BWfwwdx1NiZ/view?usp=sharing)

### 2. Prepare data

- Download the Synapse dataset from [official website](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789). Convert them to numpy format, clip within [-125, 275], normalize each 3D volume to [0, 1], and extract 2D slices from 3D volume for training while keeping the testing 3D volume in h5 format.

- Or directly use [preprocessed data](https://drive.google.com/file/d/1XjHzJageFKFN7Tg-6F2NJz2sj9hSLPK0/view?usp=sharing) provided by [TransUNet](https://github.com/Beckschen/TransUNet).

### 3. Environment

We trained our model on one NVIDIA GeForce GTX 3090 with the CUDA 11.1 and CUDNN 8.0.

- Python 3.8.13.

- PyTorch 1.8.1. 

- Please refer to 'requirements.txt' for other dependencies.

### 4. Evaluate our model trained on the Cityscapes Train set 

- Download the trained model:[link](https://drive.google.com/file/d/1vGLHOW-_ref28PC0LXSyMuW-J6QJRPLB/view?usp=sharing). On the Valuation set, this trained model reaches 78.2% and 77.9% with input size 1024x2048 and 1024x1024, respectively. Its pruned version reaches 77.9% and 77.7%, respectively.

- Put 'MFARANet_resnet_18_deep_stem_cityscapes_2GPU_train_lr_005_batch_14_200_200.pth' into this folder: './save/model/our_model_train_val'. Run the following order:

```bash
cd MFARANet-main
```

```bash
python train_val_inference_general.py --yaml_Name='MFARANet_train_valuation_Basic_Config.yaml' --train_gpu 0 --NAME_model 'MFARANet_resnet_18_deep_stem' --Marker 'Branch_1_2_3_4_Paper_Val' --load_trained_model='./save/model/our_model_train_val/MFARANet_resnet_18_deep_stem_cityscapes_2GPU_train_lr_005_batch_14_200_200.pth'
```

### 5. Evaluate our model trained on the Cityscapes Train + Valuation set 

- Download the trained model:[link](https://drive.google.com/file/d/155ygZ50a6EwGqjm6qEnZ1shGQ9J3TKxn/view?usp=sharing). On the Test set, this trained model reaches 77.3% and 77.1% with input size 1024x2048 and 1024x1024, respectively. Its pruned version reaches 77.0% and 76.8%, respectively.

- Put 'MFARANet_resnet_18_deep_stem_cityscapes_2GPU_train_lr_005_batch_14_275_274.pth' into this folder: './save/model/our_model_trainval_test'. Run the following order:

```bash
python train_val_inference_general.py --yaml_Name='MFARANet_train_valuation_Basic_Config_Trainval_Test.yaml' --train_gpu 1 --NAME_model 'MFARANet_resnet_18_deep_stem' --Marker 'Branch_1_2_3_4_Paper_TrainVal_Test' --load_trained_model='./save/model/our_model_trainval_test/MFARANet_resnet_18_deep_stem_cityscapes_2GPU_train_lr_005_batch_14_275_274.pth'
```

- Run the following order to transform the predicted segmentation map from trainID to labelID, then follow the instruction in the [official website](https://www.cityscapes-dataset.com/) of the Cityscapes to submit to their test service:

```bash
CUDA_VISIBLE_DEVICES=0 python Submit_Cityscapes_trainID_2_labelID.py --path_Name='MFARANet_resnet_18_deep_stem_Branch_1_2_3_4_Paper_TrainVal_Test_cityscapes'
```

### 5. Train/Test by yourself

```bash
cd Project_MultiTrans_V0
```

- Run the train script.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --Model_Name My_MultiTrans_V0 --branch_in_channels 128 256 512 512 1024 --branch_out_channels 256 --branch_key_channels 8 16 32 64 128 --branch_choose 1 2 3 4 --seed 1294
```

- Run the test script.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --dataset Synapse --Model_Name My_MultiTrans_V0 --branch_in_channels 128 256 512 512 1024 --branch_out_channels 256 --branch_key_channels 8 16 32 64 128 --branch_choose 1 2 3 4 --seed 1294
```

### 6. Ablation experiments on multi-branch design

Add following orders to train-script and test-script.

- Use a single Transformer branch:

```bash
--branch_choose 1   # 1 or 2 or 3 or 4
```

- Remove one of the four branches:

```bash
--branch_choose 2 3 4   # 2 3 4 or 1 3 4 or 1 2 4 or 1 2 3 
```
### 7. Ablation experiments on the design of efficient self-attention

- If_efficient_attention: use Order-Changing or not; one_kv_head: use Head-Sharing or not; share_kv: use Projection-Sharing or not:

```bash
--If_efficient_attention True --one_kv_head True --share_kv False   
```

- If you want to replace our efficient self-attention with stand self-attention, you need to train the model on 3 GPUs with halved batch sizes and base_lr:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python train.py --dataset Synapse --Model_Name My_MultiTrans_V0 --branch_in_channels 128 256 512 512 1024 --branch_out_channels 256 --branch_key_channels 8 16 32 64 128 --branch_choose 1 2 3 4 --If_efficient_attention False --n_gpu 3 --batch_size 4 --base_lr 0.005 --seed 1294
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python test.py --dataset Synapse --Model_Name My_MultiTrans_V0 --branch_in_channels 128 256 512 512 1024 --branch_out_channels 256 --branch_key_channels 8 16 32 64 128 --branch_choose 1 2 3 4 --If_efficient_attention False --n_gpu 3 --batch_size 4 --base_lr 0.005 --seed 1294
```

## Reference
* [TransUNet](https://github.com/Beckschen/TransUNet)

## Citations

```bibtex

xxx

```
