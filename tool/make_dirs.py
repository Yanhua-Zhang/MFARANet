import sys 
sys.path.append("..")
import os
from util.util import check_makedirs, get_args

# =====================================================================
yaml_path = '/home/zhangyanhua/Code_python/PSP-args-logger-multiprocessing-general/config/PASCAL_context/PASCAL_context_pspnet50.yaml'
cfg = get_args(yaml_path)
# =====================================================================
# 进行文件夹的生成
# check_makedirs(cfg['absolute_path']+cfg['FILE_NAME']+"/save/log/"+cfg['NAME'])            # 检查保存 log 的文件夹是否存在，不存在进行生成
# check_makedirs(cfg['absolute_path']+cfg['FILE_NAME']+'/save/tensorboardX/'+cfg['NAME'])   # 检查文件夹是否存在，不存在生成
# check_makedirs(cfg['absolute_path'] +cfg['FILE_NAME']+'/save/model/'+ cfg['NAME'])        # 检查保存模型的文件夹是否存在，不存在生成
# check_makedirs(cfg['absolute_path'] +cfg['FILE_NAME']+'/save/model_best/'+ cfg['NAME'])   # 检查保存模型的文件夹是否存在，不存在生成

# 注意!!!
check_makedirs(cfg['absolute_path'] +cfg['FILE_NAME']+'/config/'+ cfg['NAME_dataset']+'/'+'test')   # 检查保存模型的文件夹是否存在，不存在生成