from utils.datatool import (get_val_transformations,
                      get_train_dataset,
                      get_val_dataset)
from configs.basic_config import get_cfg
import os
from utils.datatool import __dataset_dict
import json
import random

if __name__=="__main__":
    cfg_file_list = ["./configs/EdgeMNIST/consist.yaml",
                "./configs/FMNIST/consist.yaml",
                "./configs/coil-20/consist.yaml",
                "./configs/coil-100/consist.yaml"]
    
    # for key in __dataset_dict.keys():
    #     dataclass = __dataset_dict.get(key, None)
    #     train_set = dataclass(root='MyData', train=False,
    #                         transform=get_val_transformations(), download=True)
    res_dir = "./MaskView"
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)
    for cfg_path in cfg_file_list:
        config = get_cfg(cfg_path)
        data_class = __dataset_dict.get(config.dataset.name, None)
        if data_class is None:
            raise ValueError("Dataset name error.")
        val_set = data_class(root=config.dataset.root, train=False,
                            transform=get_val_transformations(config), views=config.views)
        
        mask_view_ratio = config.train.val_mask_view_ratio
        # reproduct
        seed = config.seed
        random.seed(seed)
        num_to_select = int(len(val_set)*mask_view_ratio)
        random_indices = random.sample(range(len(val_set)), num_to_select)
        random_views = [random.randint(0, config.views-1) for _ in range(num_to_select)]
        #Write the file
        file_path = os.path.join(res_dir, config.dataset.name+".json")
        with open(file_path, "w") as file:
            json.dump({'indices':random_indices, 'views':random_views}, file)
        

            
        
        
            
        

        

        



        
        
        
        

    