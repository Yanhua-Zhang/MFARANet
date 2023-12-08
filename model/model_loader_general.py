from .model_MFARANet import MFARANet
from .model_MFARANet_Scale_Choice import MFARANet_Scale_Choice

from .model_MFARANet_Scale_Choice_STDC import MFARANet_Scale_Choice_STDC

from .model_MFARANet_Scale_Choice_DFNet import MFARANet_Scale_Choice_DFNet

from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, find_free_port, get_args, build_logger, check_makedirs

def load_model(cfg, criterion, split):
    # file_name_log = cfg['absolute_path']+cfg['FILE_NAME']+"/save/log/"+cfg['NAME_model'] +'_'+ cfg['NAME_dataset']+'/'+"record_"+ cfg['NAME_model'] +'_'+ cfg['NAME_dataset']  +"_train.log"
    # logger = build_logger(cfg['logger_name_trainval'] , file_name_log)

    if split == 'train':
        
        # -----------------------------------------------------------------------------------
        if cfg['NAME_model'].split('_')[0] == 'MFARANet':
            
            print('利用 model_loader_general 加载 model：')
            print(cfg['NAME_model'].split('_')[0])
            print('---------------------------------------------------')
            model = MFARANet(backbone_name=cfg['backbone_name'], classes=cfg['num_classes'], 
                                use_dilation=cfg['use_dilation'], use_PPM=cfg['use_PPM'], 
                                use_aux_loss=cfg['use_aux_loss'], 
                                if_use_boundary_loss = cfg['if_use_boundary_loss'],
                                criterion=criterion)
            ori_name = []
            modules_ori = []
            new_name = []
            modules_new = []
            for name, module in model.named_children():
                if name in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']:
                    ori_name.append(name)
                    modules_ori.append(module)
                elif name != 'criterion':
                    new_name.append(name)
                    modules_new.append(module)

            print('Encoder 各 layer name：')
            print(ori_name)
            print('---------------------------------------------------')
            print('Decoder 各 layer name：')
            print(new_name)
            print('---------------------------------------------------')

        elif cfg['NAME_model'].split('_')[0] == 'MFARANetScaleChoice':
            
            print('利用 model_loader_general 加载 model：')
            print(cfg['NAME_model'].split('_')[0])
            print('---------------------------------------------------')
            model = MFARANet_Scale_Choice(backbone_name=cfg['backbone_name'], classes=cfg['num_classes'], 
                                use_dilation=cfg['use_dilation'], use_PPM=cfg['use_PPM'], 
                                use_aux_loss=cfg['use_aux_loss'], 
                                ASFM_stage_choose = cfg['Branch_Choose'],
                                Dropout_Rate_CNN = cfg['Dropout_Rate_CNN'],
                                if_use_boundary_loss = cfg['if_use_boundary_loss'],
                                criterion=criterion)
            ori_name = []
            modules_ori = []
            new_name = []
            modules_new = []
            for name, module in model.named_children():
                if name in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']:
                    ori_name.append(name)
                    modules_ori.append(module)
                elif name != 'criterion':
                    new_name.append(name)
                    modules_new.append(module)

            print('Encoder 各 layer name：')
            print(ori_name)
            print('---------------------------------------------------')
            print('Decoder 各 layer name：')
            print(new_name)
            print('---------------------------------------------------')


        elif cfg['NAME_model'].split('_')[0] == 'MFARANetScaleChoiceSTDC':
            
            print('利用 model_loader_general 加载 model：')
            print(cfg['NAME_model'].split('_')[0])
            print('---------------------------------------------------')
            model = MFARANet_Scale_Choice_STDC(backbone_name=cfg['backbone_name'], classes=cfg['num_classes'], 
                                use_dilation=cfg['use_dilation'], use_PPM=cfg['use_PPM'], 
                                use_aux_loss=cfg['use_aux_loss'], 
                                ASFM_stage_choose = cfg['Branch_Choose'],
                                Dropout_Rate_CNN = cfg['Dropout_Rate_CNN'],
                                if_use_boundary_loss = cfg['if_use_boundary_loss'],
                                criterion=criterion)
            ori_name = []
            modules_ori = []
            new_name = []
            modules_new = []
            for name, module in model.named_children():
                if name in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']:
                    ori_name.append(name)
                    modules_ori.append(module)
                elif name != 'criterion':
                    new_name.append(name)
                    modules_new.append(module)

            print('Encoder 各 layer name：')
            print(ori_name)
            print('---------------------------------------------------')
            print('Decoder 各 layer name：')
            print(new_name)
            print('---------------------------------------------------')

        elif cfg['NAME_model'].split('_')[0] == 'MFARANetScaleChoiceDFNet':
            
            print('利用 model_loader_general 加载 model：')
            print(cfg['NAME_model'].split('_')[0])
            print('---------------------------------------------------')
            model = MFARANet_Scale_Choice_DFNet(backbone_name=cfg['backbone_name'], classes=cfg['num_classes'], 
                                use_dilation=cfg['use_dilation'], use_PPM=cfg['use_PPM'], 
                                use_aux_loss=cfg['use_aux_loss'], 
                                ASFM_stage_choose = cfg['Branch_Choose'],
                                Dropout_Rate_CNN = cfg['Dropout_Rate_CNN'],
                                if_use_boundary_loss = cfg['if_use_boundary_loss'],
                                criterion=criterion)
            ori_name = []
            modules_ori = []
            new_name = []
            modules_new = []
            for name, module in model.named_children():
                if name in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']:
                    ori_name.append(name)
                    modules_ori.append(module)
                elif name != 'criterion':
                    new_name.append(name)
                    modules_new.append(module)

            print('Encoder 各 layer name：')
            print(ori_name)
            print('---------------------------------------------------')
            print('Decoder 各 layer name：')
            print(new_name)
            print('---------------------------------------------------')

        else:
            raise (RuntimeError("no model found"))

# ----------------------------------------------------------------------------------------------------------------------------------------
# test 时加载的网络
# ----------------------------------------------------------------------------------------------------------------------------------------
    else:   # test 时加载的网络

        # -----------------------------------------------------------------------------------
        if cfg['NAME_model'].split('_')[0] == 'MFARANet':
            
            print('利用 model_loader_general 加载 model：')
            print(cfg['NAME_model'].split('_')[0])
            print('---------------------------------------------------')
            model = MFARANet(backbone_name=cfg['backbone_name'], classes=cfg['num_classes'], 
                                use_dilation=cfg['use_dilation'], use_PPM=cfg['use_PPM'], 
                                use_aux_loss=cfg['use_aux_loss'], 
                                if_use_boundary_loss = cfg['if_use_boundary_loss'],
                                pretrained = cfg['if_pretrain'])
            ori_name = []
            modules_ori = []
            new_name = []
            modules_new = []
            for name, module in model.named_children():
                if name in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']:
                    ori_name.append(name)
                    modules_ori.append(module)
                elif name != 'criterion':
                    new_name.append(name)
                    modules_new.append(module)

            print('Encoder 各 layer name：')
            print(ori_name)
            print('---------------------------------------------------')
            print('Decoder 各 layer name：')
            print(new_name)
            print('---------------------------------------------------')

        elif cfg['NAME_model'].split('_')[0] == 'MFARANetScaleChoice':
            
            print('利用 model_loader_general 加载 model：')
            print(cfg['NAME_model'].split('_')[0])
            print('---------------------------------------------------')
            model = MFARANet_Scale_Choice(backbone_name=cfg['backbone_name'], classes=cfg['num_classes'], 
                                use_dilation=cfg['use_dilation'], use_PPM=cfg['use_PPM'], 
                                use_aux_loss=cfg['use_aux_loss'], 
                                ASFM_stage_choose = cfg['Branch_Choose'],
                                Dropout_Rate_CNN = cfg['Dropout_Rate_CNN'],
                                if_use_boundary_loss = cfg['if_use_boundary_loss'],
                                pretrained = cfg['if_pretrain'])
            ori_name = []
            modules_ori = []
            new_name = []
            modules_new = []
            for name, module in model.named_children():
                if name in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']:
                    ori_name.append(name)
                    modules_ori.append(module)
                elif name != 'criterion':
                    new_name.append(name)
                    modules_new.append(module)

            print('Encoder 各 layer name：')
            print(ori_name)
            print('---------------------------------------------------')
            print('Decoder 各 layer name：')
            print(new_name)
            print('---------------------------------------------------')

        elif cfg['NAME_model'].split('_')[0] == 'MFARANetScaleChoiceSTDC':
            
            print('利用 model_loader_general 加载 model：')
            print(cfg['NAME_model'].split('_')[0])
            print('---------------------------------------------------')
            model = MFARANet_Scale_Choice_STDC(backbone_name=cfg['backbone_name'], classes=cfg['num_classes'], 
                                use_dilation=cfg['use_dilation'], use_PPM=cfg['use_PPM'], 
                                use_aux_loss=cfg['use_aux_loss'], 
                                ASFM_stage_choose = cfg['Branch_Choose'],
                                Dropout_Rate_CNN = cfg['Dropout_Rate_CNN'],
                                if_use_boundary_loss = cfg['if_use_boundary_loss'],
                                pretrained = cfg['if_pretrain'])
            ori_name = []
            modules_ori = []
            new_name = []
            modules_new = []
            for name, module in model.named_children():
                if name in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']:
                    ori_name.append(name)
                    modules_ori.append(module)
                elif name != 'criterion':
                    new_name.append(name)
                    modules_new.append(module)

            print('Encoder 各 layer name：')
            print(ori_name)
            print('---------------------------------------------------')
            print('Decoder 各 layer name：')
            print(new_name)
            print('---------------------------------------------------')


        elif cfg['NAME_model'].split('_')[0] == 'MFARANetScaleChoiceDFNet':
            
            print('利用 model_loader_general 加载 model：')
            print(cfg['NAME_model'].split('_')[0])
            print('---------------------------------------------------')
            model = MFARANet_Scale_Choice_DFNet(backbone_name=cfg['backbone_name'], classes=cfg['num_classes'], 
                                use_dilation=cfg['use_dilation'], use_PPM=cfg['use_PPM'], 
                                use_aux_loss=cfg['use_aux_loss'], 
                                ASFM_stage_choose = cfg['Branch_Choose'],
                                Dropout_Rate_CNN = cfg['Dropout_Rate_CNN'],
                                if_use_boundary_loss = cfg['if_use_boundary_loss'],
                                pretrained = cfg['if_pretrain'])
            ori_name = []
            modules_ori = []
            new_name = []
            modules_new = []
            for name, module in model.named_children():
                if name in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']:
                    ori_name.append(name)
                    modules_ori.append(module)
                elif name != 'criterion':
                    new_name.append(name)
                    modules_new.append(module)

            print('Encoder 各 layer name：')
            print(ori_name)
            print('---------------------------------------------------')
            print('Decoder 各 layer name：')
            print(new_name)
            print('---------------------------------------------------')


        else:
            raise (RuntimeError("no model found"))

    return model, modules_ori, modules_new