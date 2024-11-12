import argparse
import os
from collections import defaultdict
from itertools import chain
import time
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import numpy as np
import wandb
from tqdm import tqdm
import torch.distributed as dist
from torchinfo import summary
from configs.basic_config import get_cfg
from models.mrdd import MRDD
from utils import (clustering_by_representation,
                   reproducibility_setting, get_device, save_checkpoint)
from utils.datatool import (get_val_transformations,
                      get_train_dataset,
                      get_val_dataset,
                      get_mask_val)
from optimizer import get_optimizer, get_scheduler

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', '-f', type=str, help='Config File')
    args = parser.parse_args()
    return args


def init_distributed_mode():
    # set cuda device
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(
        backend='nccl' if dist.is_nccl_available() else 'gloo')


def clean_distributed():
    dist.destroy_process_group()


def smartprint(*msg):
    if LOCAL_RANK == 0 or LOCAL_RANK == -1:
        print(*msg)

# @torch.no_grad()
# def valid_by_kmeans(val_dataloader, model, use_ddp, device):
#     targets = []
#     consist_reprs = []
#     vspecific_reprs = []
#     concate_reprs = []
#     for Xs, target in val_dataloader:
#         Xs = [x.to(device) for x in Xs]
#         if use_ddp:
#             consist_repr_, vspecific_repr_, concate_repr_, _ = model.module.all_features(Xs)
#         else:
#             consist_repr_, vspecific_repr_, concate_repr_, _ = model.all_features(Xs)
#         targets.append(target)
#         consist_reprs.append(consist_repr_.detach().cpu())
#         vspecific_reprs.append(vspecific_repr_.detach().cpu())
#         concate_reprs.append(concate_repr_.detach().cpu())
#     targets = torch.concat(targets, dim=-1).numpy()
#     consist_reprs = torch.vstack(consist_reprs).detach().cpu().numpy()
#     vspecific_reprs = torch.vstack(vspecific_reprs).detach().cpu().numpy()
#     concate_reprs = torch.vstack(concate_reprs).detach().cpu().numpy()
#     result = {}
#     acc, nmi, ari, _, p, fscore = clustering_by_representation(consist_reprs, targets)
#     result['consist-acc'] = acc
#     result['consist-nmi'] = nmi
#     result['consist-ari'] = ari
#     result['consist-p'] = p
#     result['consist-fscore'] = fscore
#
#     acc, nmi, ari, _, p, fscore = clustering_by_representation(vspecific_reprs, targets)
#     result['vspec-acc'] = acc
#     result['vspec-nmi'] = nmi
#     result['vspec-ari'] = ari
#     result['vspec-p'] = p
#     result['vspec-fscore'] = fscore
#
#     acc, nmi, ari, _, p, fscore = clustering_by_representation(concate_reprs, targets)
#     result['cat-acc'] = acc
#     result['cat-nmi'] = nmi
#     result['cat-ari'] = ari
#     result['cat-p'] = p
#     result['cat-fscore'] = fscore
#     return result
@torch.no_grad()
def valid_by_kmeans(val_dataloader, model, use_ddp, device, noise=False):
    targets = []
    consist_reprs = []
    vspecific_reprs = defaultdict(list)
    concate_reprs = defaultdict(list)
    for Xs, target in val_dataloader:
        if noise:
            Xs = [torch.clip(x+torch.randn_like(x), 0, 1).to(device) for x in Xs]
        else:
            # print(Xs[0].shape)
            Xs = [x.to(device) for x in Xs]
        if use_ddp:
            consist_repr_, vspecific_repr_, concate_repr_ = model.module.all_features(Xs)
        else:
            consist_repr_, vspecific_repr_, concate_repr_ = model.all_features(Xs)   # Tensor, list, list

        targets.append(target)
        consist_reprs.append(consist_repr_.detach().cpu())
        # vspecific_reprs.append(vspecific_repr_.detach().cpu())
        # concate_reprs.append(concate_repr_.detach().cpu())
        for i, (si, c_si) in enumerate(zip(vspecific_repr_, concate_repr_)):
            vspecific_reprs[f"s{i}"].append(si.detach().cpu())
            concate_reprs[f"c+s{i}"].append(c_si.detach().cpu())

    targets = torch.concat(targets, dim=-1).numpy()
    consist_reprs = torch.vstack(consist_reprs).detach().cpu().numpy()
    # vspecific_reprs = torch.vstack(vspecific_reprs).detach().cpu().numpy()
    # concate_reprs = torch.vstack(concate_reprs).detach().cpu().numpy()
    result = {}
    acc, nmi, ari, _, p, fscore = clustering_by_representation(consist_reprs, targets)
    result['consist-acc'] = acc
    result['consist-nmi'] = nmi
    result['consist-ari'] = ari
    result['consist-p'] = p
    result['consist-fscore'] = fscore

    for key, spe_repr in vspecific_reprs.items():
        spe_repr = torch.vstack(spe_repr).detach().cpu().numpy()
        acc, nmi, ari, _, p, fscore = clustering_by_representation(spe_repr, targets)
        result[f'{key}-acc'] = acc
        result[f'{key}-nmi'] = nmi
        result[f'{key}-ari'] = ari
        result[f'{key}-p'] = p
        result[f'{key}-fscore'] = fscore
    for key, cat_repr in concate_reprs.items():
        cat_repr = torch.vstack(cat_repr).detach().cpu().numpy()
        acc, nmi, ari, _, p, fscore = clustering_by_representation(cat_repr, targets)
        result[f'{key}-acc'] = acc
        result[f'{key}-nmi'] = nmi
        result[f'{key}-ari'] = ari
        result[f'{key}-p'] = p
        result[f'{key}-fscore'] = fscore
    return result

def train_a_epoch(args, train_dataloader, model, epoch, device, optimizers, lr):
    losses = defaultdict(list)

    if args.train.use_ddp:
        model.module.train()
        model.module.consis_enc.eval()
    else:
        model.train()
        model.consis_enc.eval()
    if args.verbose and (LOCAL_RANK == 0 or LOCAL_RANK == -1):
        pbar = tqdm(train_dataloader, ncols=0, unit=" batch")
        
    for Xs, _ in tqdm(train_dataloader):
        
        Xs = [x.to(device) for x in Xs]
        
        if args.train.use_ddp:
            loss, loss_part = model.module.get_loss(Xs)
        else:
            loss, loss_part = model.get_loss(Xs)
        for idx in range(len(loss)):  
            optimizer = optimizers[idx]
            optimizer.zero_grad()
            loss[idx].backward()
            optimizer.step()

        for k, v in loss_part.items():
            losses[k].append(v)  
        
        show_losses = {k: np.mean(v) for k, v in losses.items()}
        
        if args.verbose and (LOCAL_RANK == 0 or LOCAL_RANK == -1):
            loss_str = ', '.join([f'{k}:{v:.4f}' for k, v in show_losses.items()])
            pbar.set_description(f"Training | epoch: {epoch}, lr: {lr:.4f}, {loss_str}")
            pbar.update()
    
    if args.verbose and (LOCAL_RANK == 0 or LOCAL_RANK == -1):
        pbar.close()
        
    return show_losses


def main():
    # Load arguments.
    args = parse_args()
    config = get_cfg(args.config_file)
    
    use_wandb = config.wandb
    use_ddp = config.train.use_ddp
    seed = config.seed
    runtimes = config.runtimes
    result_dir = os.path.join(config.train.log_dir, f'disent-m{config.train.masked_ratio}-mv{config.train.mask_view_ratio if config.train.mask_view else 0.0}-c{config.consistency.c_dim}-v{config.vspecific.v_dim}-{seed}')
    os.makedirs(result_dir, exist_ok=True)
    
    
    if use_ddp:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
            [str(i) for i in config.train.devices])
    
    device = get_device(config, LOCAL_RANK)
    print(f'Use: {device}')

    if use_ddp:
        init_distributed_mode()

    consistency_encoder_path = config.consistency.consistency_encoder_path
    evaluate_intervals = config.train.evaluate

    # for record each running.
    running_loggers = {}
    for r in range(runtimes):
    
        sub_logger = defaultdict(list)
        
        finalmodel_path = os.path.join(result_dir, f'final_model-{seed}.pth')

        # For reproducibility
        reproducibility_setting(seed)

        val_transformations = get_val_transformations(config)
        train_dataset = get_train_dataset(config, val_transformations)
        
        if use_ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
        train_dataloader = DataLoader(train_dataset,
                                    num_workers=config.train.num_workers,
                                    batch_size=config.train.batch_size // WORLD_SIZE,
                                    sampler=train_sampler if use_ddp else None,
                                    shuffle=False if use_ddp else True,
                                    pin_memory=True,
                                    drop_last=True)
        # Only evaluation at the first device.
        if LOCAL_RANK == 0 or LOCAL_RANK == -1:

            mask_val_dataset = get_mask_val(config, val_transformations)
            mask_val_dataloader = DataLoader(mask_val_dataset,
                                             batch_size=config.train.batch_size // WORLD_SIZE,
                                             num_workers=config.train.num_workers,
                                             shuffle=False,
                                             drop_last=False,
                                             pin_memory=True)
            val_dataset = get_val_dataset(config, val_transformations)
            val_dataloader = DataLoader(val_dataset,
                                        batch_size=config.train.batch_size // WORLD_SIZE,
                                        num_workers=config.train.num_workers,
                                        shuffle=False,
                                        drop_last=False,
                                        pin_memory=True)

            smartprint('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))
        
            dl = DataLoader(val_dataset, 16, shuffle=True)
            recon_samples = next(iter(dl))[0]
            recon_samples = [x.to(device, non_blocking=True) for x in recon_samples]
        
 
        model = MRDD(config, consistency_encoder_path=consistency_encoder_path, device=device)
        if LOCAL_RANK == 0 or LOCAL_RANK == -1:
            summary(model)
        smartprint(f"Consistency model loaded! ")
        
        
        optimizers = [get_optimizer(chain(*model.get_vsepcific_params(vid)), config.train.lr, config.train.optim) for vid in range(config.views)]

        if config.train.scheduler == 'constant':
            no_scheduler = True
        else:
            no_scheduler = False
            schedulers = [get_scheduler(config, optimizers[vid]) for vid in range(config.views)]

        start_epoch = 0
        model = model.to(device)

        if use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[LOCAL_RANK],
                output_device=LOCAL_RANK,
                find_unused_parameters=True,
                broadcast_buffers=False
            )
        
        if use_wandb and (LOCAL_RANK == 0 or LOCAL_RANK == -1):
            wandb.init(project=config.project_name, config=config,
                    name=f"{config.experiment_name}-disent-m{config.train.masked_ratio}-mv{config.train.mask_view_ratio if config.train.mask_view else 0.0}-c{config.consistency.c_dim}-v{config.vspecific.v_dim}-{seed}")
            # wandb.watch(model, log='all', log_graph=True, log_freq=15)
        
        for epoch in range(start_epoch, config.train.epochs):

            lr = optimizers[0].param_groups[0]['lr']
            
            # Train
            if use_ddp:
                train_dataloader.sampler.set_epoch(epoch)
            losses = train_a_epoch(config, train_dataloader, model, epoch, device, optimizers, lr)
            if use_wandb and (LOCAL_RANK == 0 or LOCAL_RANK == -1):
                wandb.log(losses, step=epoch)
                    
            if not config.verbose:
                smartprint(f"[Training {epoch}/{config.train.epochs}]", ', '.join([f'{k}:{v:.4f}' for k, v in losses.items()]))
            
            for k, v in losses.items():
                sub_logger[k].append(v)    
                    
            if not no_scheduler:
                for vid in range(config.views):
      
                    schedulers[vid].step()

            if LOCAL_RANK == 0 or LOCAL_RANK == -1:
                if epoch % evaluate_intervals == 0:
                    if config.train.use_ddp:
                        model.module.eval()
                    else:
                        model.eval()

                        # validate on full modal
                        kmeans_result = valid_by_kmeans(val_dataloader=val_dataloader,
                                                        model=model,
                                                        device=device,
                                                        use_ddp=use_ddp)
                        print(f"[Evaluation {epoch}/{config.train.epochs}]",
                              ', '.join([f'{k}:{v:.4f}' for k, v in kmeans_result.items()]))
                        if use_wandb:
                            wandb.log(kmeans_result, step=epoch)

                        # validate on modal missing
                        kmeans_result = valid_by_kmeans(val_dataloader=mask_val_dataloader,
                                                        model=model,
                                                        device=device,
                                                        use_ddp=use_ddp)
                        print(f"[Modal missing]",
                              ', '.join([f'{k}:{v:.4f}' for k, v in kmeans_result.items()]))
                        if use_wandb:
                            for k, v in kmeans_result.items():
                                wandb.log({k + "(modal missing)": v}, step=epoch)

                        # validate on full modal with Gaussian Noise
                        kmeans_result = valid_by_kmeans(val_dataloader=val_dataloader,
                                                        model=model,
                                                        device=device,
                                                        use_ddp=use_ddp,
                                                        noise=True)
                        print(f"[Data with Noise]",
                              ', '.join([f'{k}:{v:.4f}' for k, v in kmeans_result.items()]))
                        if use_wandb:
                            for k, v in kmeans_result.items():
                                wandb.log({k + "(with noise)": v}, step=epoch)
                    
                        
                    if use_wandb:
                        wandb.log(kmeans_result, step=epoch)
                
        # update seed.        
        running_loggers[f'r{r+1}-{seed}'] = sub_logger
        seed = torch.randint(1, 9999, (1, )).item()    
        if LOCAL_RANK == 0 or LOCAL_RANK == -1:
            if config.train.use_ddp:
                model.module.eval()
                # Save final model
                torch.save(model.module.state_dict(), finalmodel_path) 
            else:
                model.eval()
                # Save final model
                torch.save(model.state_dict(), finalmodel_path)


                     
    if LOCAL_RANK == 0 or LOCAL_RANK == -1:            
        torch.save(running_loggers, os.path.join(
            result_dir, 'loggers.pkl'))
        
    if use_ddp:
        clean_distributed()


@torch.no_grad()
def sampling(model, samples_num, device, use_ddp):
    """
    Sampling from conditional vaes
    """
    if use_ddp:
        outs = model.module.sampling(samples_num, device=device)
    else:
        outs = model.sampling(samples_num, device=device)
    sample_grid = make_grid(outs.detach().cpu())
    return sample_grid


@torch.no_grad()
def reconstruction(model, original, use_ddp):
    if use_ddp:
        recons = model.module(original)
    else:
        recons = model(original)
    grid = []
    for x, r in zip(original, recons):
        grid.append(torch.cat([x, r]).detach().cpu())
    grid = make_grid(torch.cat(grid).detach().cpu())
    return grid
    

    
if __name__ == '__main__':
    main()
    