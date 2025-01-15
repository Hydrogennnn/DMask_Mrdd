import warnings
warnings.filterwarnings('ignore')
import argparse
import os
from collections import defaultdict

import torch
import random
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import numpy as np
import json

from scipy.optimize import linear_sum_assignment
from configs.basic_config import get_cfg
from models.mrdd import MRDD
from sklearn import metrics
from utils.datatool import (get_val_transformations,
                            get_train_dataset,
                            get_mask_train_dataset,
                            add_sp_noise)
from utils.visualization import plot_scatter

def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def clustering_metric(y_true, y_pred, decimals=4):
    """Get clustering metric"""

    # ACC
    acc = clustering_accuracy(y_true, y_pred)
    acc = np.round(acc, decimals)

    # NMI
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    nmi = np.round(nmi, decimals)
    # ARI
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    ari = np.round(ari, decimals)

    return acc, nmi, ari


def classification_metric(y_true, y_pred, average='macro', verbose=True, decimals=4):
    """Get classification metric"""

    # ACC
    accuracy = metrics.accuracy_score(y_true, y_pred)
    accuracy = np.round(accuracy, decimals)

    # precision
    precision = metrics.precision_score(y_true, y_pred, average=average)
    precision = np.round(precision, decimals)

    # F-score
    f_score = metrics.f1_score(y_true, y_pred, average=average)
    f_score = np.round(f_score, decimals)

    return accuracy, precision, f_score

@torch.no_grad()
def reconstruction(model, original):
    vspecific_recons = model(original)
    consist_recons, _ = model.consis_enc(original)
    grid = []
    for x, r in zip(original, vspecific_recons):
        grid.append(torch.cat([x, r]).detach().cpu())
    vspec_grid = make_grid(torch.cat(grid).detach().cpu())

    grid = []
    for x, r in zip(original, consist_recons):
        grid.append(torch.cat([x, r]).detach().cpu())
    consist_grid = make_grid(torch.cat(grid).detach().cpu())
    return consist_grid, vspec_grid


@torch.no_grad()
def extract_features(val_dataloader, model, device, noise_prob=None):
    targets = []
    consist_reprs = []
    vspecific_reprs = []
    concate_reprs = []
    all_vs = []
    for Xs, target in val_dataloader:
        if noise_prob:
            Xs = [add_sp_noise(x, noise_prob).to(device) for x in Xs]
        else:
            Xs = [x.to(device) for x in Xs]
        consist_repr_, vspecific_repr_, concate_repr_, all_v = model.all_features(Xs)
        targets.append(target)
        consist_reprs.append(consist_repr_.detach().cpu())
        vspecific_reprs.append(vspecific_repr_.detach().cpu())
        concate_reprs.append(concate_repr_.detach().cpu())
        all_vs.append(all_v)
    targets = torch.concat(targets, dim=-1).numpy()
    consist_reprs = torch.vstack(consist_reprs).detach().cpu()
    vspecific_reprs = torch.vstack(vspecific_reprs).detach().cpu()
    concate_reprs = torch.vstack(concate_reprs).detach().cpu()
    all_vs = torch.vstack(all_vs).detach().cpu()
    return consist_reprs, vspecific_reprs, concate_reprs, all_vs, targets

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', '-f', type=str, help='Config File')
    args = parser.parse_args()
    return args


def generate_missing(m_ratio, tot_nums, dataset_name, num_views):
    res_dir = os.path.join("MaskView", dataset_name, str(m_ratio))
    file_path = os.path.join(res_dir, "train.json")
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)
    # Write the file
    random.seed = 42
    num_to_select = int(tot_nums * m_ratio)
    random_indices = random.sample(range(tot_nums), num_to_select)
    random_views = [random.randint(0, num_views - 1) for _ in range(num_to_select)]
    print('name:', dataset_name, 'len:', tot_nums, 'num_views:', num_views)
    with open(file_path, "w") as file:
        json.dump({'indices': random_indices, 'views': random_views}, file)

def main():
    # Load arguments.
    args = parse_args()
    config = get_cfg(args.config_file)
    device = torch.device(f'cuda:{config.train.devices[0]}')


    val_transformations = get_val_transformations(config)
    train_set = get_train_dataset(config, val_transformations)
    train_dataloader = DataLoader(train_set,
                                num_workers=0,
                                batch_size=config.train.batch_size,
                                sampler=None,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False)


    print(f"Contain {len(train_set)} eval samples")


    run_times = 10
    n_clusters = config.dataset.class_num
    need_classification = True
    evaluation = True
    visualization = False
    pic_format = 'pdf'
    model_path = config.eval.model_path
    model = MRDD(config, consistency_encoder_path=None, device=device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # summary(model)

    model = model.to(device)
    print(f'Use: {device}')

    model.eval()
    eval_res = defaultdict(list)
    res_dir = "eval_res"
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)
    res_path = os.path.join(res_dir, config.dataset.name + ".json")

    for i in range(0, 10):
        print(f"[Evaluation on {i / 10}modal missing]")
        generate_missing(m_ratio=i / 10, tot_nums=len(train_set), dataset_name=config.dataset.name,
                         num_views=config.views)
        mask_train_set = get_mask_train_dataset(config, val_transformations, m_ratio=i / 10)
        mask_train_dataloader = DataLoader(mask_train_set,
                                           num_workers=0,
                                           batch_size=config.train.batch_size,
                                           sampler=None,
                                           shuffle=False,
                                           pin_memory=True,
                                           drop_last=False)

        consistency, vspecific, concate, all_concate, labels = extract_features(mask_train_dataloader, model, device)
        print('eval on concate...')
        cluster_acc, _, _, cls_acc, _, _ = report(run_times, n_clusters, need_classification, labels, concate)
        eval_res["cluster-missing-mean"].append(np.mean(cluster_acc))
        eval_res["cluster-missing-std"].append(np.std(cluster_acc))
        eval_res["cls-missing-mean"].append(np.mean(cls_acc))
        eval_res["cls-missing-std"].append(np.std(cls_acc))

    for i in range(0, 10):
        print(f"[Evaluation on {i / 10} Salt-Pepper noise]")
        consistency, vspecific, concate, all_concate, labels = extract_features(train_dataloader, model, device,
                                                                                noise_prob=i / 10)
        print('eval on concate...')
        cluster_acc, _, _, cls_acc, _, _ = report(run_times, n_clusters, need_classification, labels, concate)
        eval_res["cluster-noise-mean"].append(np.mean(cluster_acc))
        eval_res["cluster-noise-std"].append(np.std(cluster_acc))
        eval_res["cls-noise-mean"].append(np.mean(cls_acc))
        eval_res["cls-noise-std"].append(np.std(cls_acc))

    with open(res_path, "w") as file:
        json.dump(eval_res, file)




def report(run_times, n_clusters, need_classification, labels, z):
    cluster_acc = []
    cluster_nmi = []
    cluster_ari = []
    cls_acc = []
    cls_p = []
    cls_fs = []

    for run in range(run_times):
        km = KMeans(n_clusters=n_clusters, n_init='auto')
        preds = km.fit_predict(z)
        acc, nmi, ari = clustering_metric(labels, preds)
        cluster_acc.append(acc)
        cluster_nmi.append(nmi)
        cluster_ari.append(ari)

        if need_classification:
            X_train, X_test, y_train, y_test = train_test_split(z, labels, test_size=0.2)
            svc = SVC()
            svc.fit(X_train, y_train)
            preds = svc.predict(X_test)
            accuracy, precision, f_score = classification_metric(y_test, preds)
            cls_acc.append(accuracy)
            cls_p.append(precision)
            cls_fs.append(f_score)

    print(f'[Clustering] acc: {np.mean(cluster_acc):.4f} ({np.std(cluster_acc):.4f}) | nmi: {np.mean(cluster_nmi):.4f} ({np.std(cluster_nmi):.4f}) \
    | ari: {np.mean(cluster_ari):.4f} ({np.std(cluster_ari):.4f})')

    if need_classification:
        print(f'[Classification] acc: {np.mean(cls_acc):.4f} ({np.std(cls_acc):.4f}) | fscore: {np.mean(cls_fs):.4f} ({np.std(cls_fs):.4f}) \
    | p: {np.mean(cls_p):.4f} ({np.std(cls_p):.4f}) ')

    return cluster_acc,cluster_nmi,cluster_ari,cls_acc,cls_p,cls_fs



if __name__ == '__main__':
    main()