import torch
from train_consistency import get_cfg, valid_by_kmeans
from utils.datatool import (get_val_transformations,
                            get_mask_val,
                            get_val_dataset)
from utils import (clustering_by_representation,
                   reproducibility_setting,
                   get_device)
from torch.utils.data import DataLoader
import os
from models.consistency_models import ConsistencyAE
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))


if __name__=='__main__':
    cfg_path = "./configs/EdgeMNIST/consist.yaml"
    model_path = "./experiments/emnist/final_model-3407.pth"
    config = get_cfg(cfg_path)
    device = get_device(config, LOCAL_RANK)
    print(f"Use device:{device}")
    useddp = config.train.use_ddp
    seed = config.seed
    reproducibility_setting(seed)
    val_transformations = get_val_transformations(config)

    if config.train.val_mask_view:
        val_dataset = get_mask_val(config, val_transformations)
    else:
        val_dataset = get_val_dataset(config, val_transformations)

    print(f"contains {len(val_dataset)} val samples")

    val_dataloader = DataLoader(val_dataset,
                                batch_size=config.train.batch_size // WORLD_SIZE,
                                num_workers=config.train.num_workers,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)

    model = ConsistencyAE(basic_hidden_dim=config.consistency.basic_hidden_dim,
                          c_dim=config.consistency.c_dim,
                          continous=config.consistency.continous,
                          in_channel=config.consistency.in_channel,
                          num_res_blocks=config.consistency.num_res_blocks,
                          ch_mult=config.consistency.ch_mult,
                          block_size=config.consistency.block_size,
                          temperature=config.consistency.temperature,
                          latent_ch=config.consistency.latent_ch,
                          kld_weight=config.consistency.kld_weight,
                          views=config.views,
                          categorical_dim=config.dataset.class_num
                          )
    model.load_state_dict(torch.load(model_path, map_location="cpu"),
                          strict=False)
    model = model.to(device)
    result = valid_by_kmeans(val_dataloader, model, useddp, device, config)
    for k, v in result.items():
        print(f"{k}:{v},")

