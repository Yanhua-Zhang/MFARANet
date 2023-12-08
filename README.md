# MultiTrans
This repository includes the official project of our paper submitted to IEEE-EMBS International Conference on Biomedical and Health Informatics (BHI’23). Title: "MultiTrans: Multi-Branch Transformer Network for Medical Image Segmentation".

## Usage

### 0. To be noted:

- We will gradually optimize the code to make it more readable and standardized.

- If you have any suggestions for improvement or encounter any issues while using this code, please feel free to contact me: zhangyanhua@mail.nwpu.edu.cn

- When loading pre-trained Resnet models and the preprocessed dataset, we use absolute paths in our code. So you can put the pre-trained models and the dataset under any path. Then, please modify their file paths in the code. The position that needs to be modified in the code can be easily found through the error message : ).

### 1. Download pre-trained Resnet models

Download the pre-trained Resnet models and the dataset under any path. Then, modify their file paths in the code.

- resnet50-deep-stem:[link](https://drive.google.com/file/d/1OktRGqZ15dIyB2YTySLfOVtprerHgbef/view?usp=sharing)

- resnet50:[link](https://drive.google.com/file/d/1fUAuRfewRpaS5mFX_IQqrE2syEn9PXrv/view?usp=sharing)

- resnet34:[link](https://drive.google.com/file/d/18Erx_ISMt1XMjJlgl4SQsr-iMvcN-7bZ/view?usp=sharing)

- resnet18-deep-stem:[link](https://drive.google.com/file/d/1q1VBV37acIte0GynoS054BWfwwdx1NiZ/view?usp=sharing)

- resnet18:[link](https://drive.google.com/file/d/1LCybGjJ_d-nALvciBBkZil_XfO-7ptAE/view?usp=sharing)

### 2. Prepare data

- Download the Synapse dataset from [official website](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789). Convert them to numpy format, clip within [-125, 275], normalize each 3D volume to [0, 1], and extract 2D slices from 3D volume for training while keeping the testing 3D volume in h5 format.

- Or directly use [preprocessed data](https://drive.google.com/file/d/1XjHzJageFKFN7Tg-6F2NJz2sj9hSLPK0/view?usp=sharing) provided by [TransUNet](https://github.com/Beckschen/TransUNet).

### 3. Environment

We trained our model on one NVIDIA GeForce GTX 3090 with the CUDA 11.1 and CUDNN 8.0.

- Python 3.8.13.

- PyTorch 1.8.1. 

- Please refer to 'requirements.txt' for other dependencies.

### 4. Test our trained model 

- Download the trained model:[link](https://drive.google.com/file/d/1HXqO9r_wmfIHzg0l0q8V5EC1cVyl-HCu/view?usp=sharing). This trained model reached 82.30% DSC and 21.10 mm HD on the Synapse dataset, without using deep supervision and sophisticated data augmentation methods. 

- Put 'epoch_149.pth' into this file: 'Results\model_Trained\My_MultiTrans_V0_Synapse224\Model\My_MultiTrans_V0_pretrain_resnet50_Deep_V0_epo150_bs24_224_s1294'. Run the following order:

```bash
cd Project_MultiTrans_V0
```

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --dataset Synapse --Model_Name My_MultiTrans_V0 --branch_in_channels 128 256 512 512 1024 --branch_out_channels 256 --branch_key_channels 8 16 32 64 128 --branch_choose 1 2 3 4 --seed 1294
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
