import yaml


def read_config(file_path):
    with open(file_path, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    cfg={}
    for key in cfg_from_file:
        # print(key)
        # items()把字典中的每对key和value组成一个元组，并把这些元祖放在列表中返回
        if type(cfg_from_file[key]) == dict:
            for k, v in cfg_from_file[key].items():
                # print(k)
                # print(v)
                cfg[k] = v
        else:
            cfg[key] = cfg_from_file[key]
    return cfg