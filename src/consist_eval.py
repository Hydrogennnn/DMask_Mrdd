import torch

from utils.datatool import (get_val_transformations,
                            get_mask_val,
                            get_val_dataset)
from utils import (clustering_by_representation,
                   reproducibility_setting,
                   get_device)
from torch.utils.data import DataLoader
from configs.basic_config import get_cfg

import os
from sklearn.model_selection import train_test_split
# from models.mrdd import MRDD
from sklearn.svm import SVC
# from eval import classification_metric
from models.consistency_models import ConsistencyAE
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))



@torch.no_grad()
def c_valid_by_kmeans(val_dataloader, model, use_ddp, device, config):
    targets = []
    consist_reprs = []
    # Extract features
    for Xs, target in val_dataloader:
        Xs = [x.to(device) for x in Xs]
        if use_ddp:
            consist_repr_ = model.module.consistency_features(Xs)
        else:
            consist_repr_ = model.consistency_features(Xs)
        targets.append(target)
        consist_reprs.append(consist_repr_.detach().cpu())
        
    targets = torch.concat(targets, dim=-1).numpy()
    consist_reprs = torch.vstack(consist_reprs).detach().cpu().numpy()
    #Clustering
    result = {}
    acc, nmi, ari, _, p, fscore = clustering_by_representation(consist_reprs, targets)
    result['consist-acc'] = acc
    result['consist-nmi'] = nmi
    result['consist-ari'] = ari
    result['consist-p'] = p
    result['consist-fscore'] = fscore

    # Classification
    X_train, X_test, y_train, y_test = train_test_split(consist_reprs, targets, test_size=0.2)
    svc = SVC()
    svc.fit(X_train, y_train)
    preds = svc.predict(X_test)
    accuracy, precision, f_score = classification_metric(y_test, preds)
    result['consist-cls_acc'] = accuracy
    result['consist-cls_precision'] = precision
    result['consist-cls_f_score'] = f_score
    return result



@torch.no_grad()
def d_valid_by_kmeans(val_dataloader, model, use_ddp, device, config):
    targets = []
    consist_reprs = []
    vspecific_reprs = []
    concate_reprs = []
    for Xs, target in val_dataloader:
        Xs = [x.to(device) for x in Xs]
        if use_ddp:
            consist_repr_, vspecific_repr_, concate_repr_, _ = model.module.all_features(Xs)
        else:
            consist_repr_, vspecific_repr_, concate_repr_, _ = model.all_features(Xs)
        targets.append(target)
        consist_reprs.append(consist_repr_.detach().cpu())
        vspecific_reprs.append(vspecific_repr_.detach().cpu())
        concate_reprs.append(concate_repr_.detach().cpu())
    targets = torch.concat(targets, dim=-1).numpy()
    consist_reprs = torch.vstack(consist_reprs).detach().cpu().numpy()
    vspecific_reprs = torch.vstack(vspecific_reprs).detach().cpu().numpy()
    concate_reprs = torch.vstack(concate_reprs).detach().cpu().numpy()
    result = {}
    acc, nmi, ari, _, p, fscore = clustering_by_representation(consist_reprs, targets)
    result['consist-acc'] = acc
    result['consist-nmi'] = nmi
    result['consist-ari'] = ari
    result['consist-p'] = p
    result['consist-fscore'] = fscore
    
    acc, nmi, ari, _, p, fscore = clustering_by_representation(vspecific_reprs, targets)
    result['vspec-acc'] = acc
    result['vspec-nmi'] = nmi
    result['vspec-ari'] = ari
    result['vspec-p'] = p
    result['vspec-fscore'] = fscore
    
    acc, nmi, ari, _, p, fscore = clustering_by_representation(concate_reprs, targets)
    result['cat-acc'] = acc
    result['cat-nmi'] = nmi
    result['cat-ari'] = ari
    result['cat-p'] = p
    result['cat-fscore'] = fscore
    return result


def Myvalid(config, model_path, val_dataloader):
    device = get_device(config, LOCAL_RANK)
    print(f"Use device:{device}")
    useddp = config.train.use_ddp
    # val_transformations = get_val_transformations(config)

    # if config.train.val_mask_view:
    #     val_dataset = get_mask_val(config, val_transformations)
    # else:
    #     val_dataset = get_val_dataset(config, val_transformations)

    # print(f"contains {len(val_dataset)} val samples")

    # val_dataloader = DataLoader(val_dataset,
    #                             batch_size=config.train.batch_size // WORLD_SIZE,
    #                             num_workers=config.train.num_workers,
    #                             shuffle=False,
    #                             drop_last=False,
    #                             pin_memory=True)

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
    model.eval()
    result = c_valid_by_kmeans(val_dataloader, model, useddp, device, config)
    print(f"Model_Load_Evaluation]", ', '.join([f'{k}:{v:.4f}' for k, v in result.items()]))

    model = MRDD(config, consistency_encoder_path=model_path, device=device)
    model = model.to(device)
    model.eval()
    result = d_valid_by_kmeans(val_dataloader,model,useddp, device, config)
    print(f"Model_Load_Evaluation]", ', '.join([f'{k}:{v:.4f}' for k, v in result.items()]))
    
    

def Mydebug(model, device, config):
    val_transformations = get_val_transformations(config)
    if config.train.val_mask_view:
        val_dataset = get_mask_val(config, val_transformations)
    else:
        val_dataset = get_val_dataset(config, val_transformations)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=config.train.batch_size // WORLD_SIZE,
                                num_workers=config.train.num_workers,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)
    model = model.to(device)
    model.eval()
    result = d_valid_by_kmeans(val_dataloader,model,False, device, config)
    print(f"Model_Load_Evaluation]", ', '.join([f'{k}:{v:.4f}' for k, v in result.items()]))
    



if __name__=='__main__':
    cfg_path = "./configs/EdgeMNIST/disent.yaml"
    model_path = "./experiments/emnist/consist-c10-m0.7-mv0.3/final_model-3407.pth"
    config = get_cfg(cfg_path)
    val_transformations = get_val_transformations(config)

    if config.train.val_mask_view:
        val_dataset = get_mask_val(config, val_transformations)
    else:
        val_dataset = get_val_dataset(config, val_transformations)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=config.train.batch_size // WORLD_SIZE,
                                num_workers=config.train.num_workers,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)
    Myvalid(config, model_path, val_dataloader)
