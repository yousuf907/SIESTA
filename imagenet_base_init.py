import numpy as np
import time
from collections import defaultdict
import faiss
import torch
import os
import utils_imagenet as utils_imagenet
import utils as utils
from retrieve_any_layer import ModelWrapper
from utils import build_classifier
import functools

def extract_features(model, data_loader, data_len, num_channels=512, spatial_feat_dim=7):
    model.eval()
    model.cuda()
    # allocate space for features and labels
    features_data = np.empty((data_len, num_channels, spatial_feat_dim, spatial_feat_dim), dtype=np.float32)
    labels_data = np.empty((data_len, 1), dtype=np.int)
    item_ixs_data = np.empty((data_len, 1), dtype=np.int)

    # put features and labels into arrays
    start_ix = 0
    for batch_ix, (batch_x, batch_y, batch_item_ixs) in enumerate(data_loader):
        batch_feats = model(batch_x.cuda())
        end_ix = start_ix + len(batch_feats)
        features_data[start_ix:end_ix] = batch_feats.cpu().numpy()
        labels_data[start_ix:end_ix] = np.atleast_2d(batch_y.numpy().astype(np.int)).transpose()
        item_ixs_data[start_ix:end_ix] = np.atleast_2d(batch_item_ixs.numpy().astype(np.int)).transpose()
        start_ix = end_ix
    return features_data, labels_data, item_ixs_data


def extract_base_init_features(imagenet_path, label_dir, extract_features_from, classifier_ckpt, arch,
                               max_class, num_channels, spatial_feat_dim, batch_size=128):
    core_model = build_classifier(arch, classifier_ckpt, num_classes=None)

    model = ModelWrapper(core_model, output_layer_names=[extract_features_from], return_single=True)

    base_train_loader = utils_imagenet.get_imagenet_data_loader(imagenet_path + '/train', label_dir, split='train',
        batch_size=batch_size, shuffle=False, min_class=0, max_class=max_class, return_item_ix=True)

    base_train_features, base_train_labels, base_item_ixs = extract_features(model, base_train_loader,
        len(base_train_loader.dataset), num_channels=num_channels, spatial_feat_dim=spatial_feat_dim)
    return base_train_features, base_train_labels, base_item_ixs
    
    
def fit_opq(feats_base_init, labels_base_init, item_ix_base_init, num_channels, spatial_feat_dim, 
    num_codebooks, codebook_size, batch_size=128, counter=utils.Counter()):
    start = time.time()
    train_data_base_init = np.transpose(feats_base_init, (0, 2, 3, 1))
    train_data_base_init = np.reshape(train_data_base_init, (-1, num_channels))
    num_samples = len(train_data_base_init)
    nbits = int(np.log2(codebook_size))
    
    ## train opq
    print('Training Optimized Product Quantizer..')
    pq = faiss.ProductQuantizer(num_channels, num_codebooks, nbits) # 80, 8, 8
    opq = faiss.OPQMatrix(num_channels, num_codebooks) # 80, 8
    opq.pq = pq
    opq.niter = 500 # optimum, higher than this provides same performance, default value=50
    #opq.verbose = True
    opq.train(np.ascontiguousarray(train_data_base_init, dtype=np.float32))
    train_data_base_init = opq.apply_py(np.ascontiguousarray(train_data_base_init, dtype=np.float32))
    pq.train(train_data_base_init)
    print("Completed in {} secs".format(time.time() - start))
    del train_data_base_init

    print('\nEncoding and Storing Base Init Codes using OPQ')
    start_time = time.time()
    latent_dict = {}
    class_id_to_item_ix_dict = defaultdict(list)
    rehearsal_ixs = []
    mb = min(batch_size, num_samples)
    for i in range(0, num_samples, mb):
        start = i
        end = min(start + mb, num_samples)
        data_batch = feats_base_init[start:end]
        batch_labels = labels_base_init[start:end]
        batch_item_ixs = item_ix_base_init[start:end]
        data_batch = np.transpose(data_batch, (0, 2, 3, 1)) # Nx14x14x80
        data_batch = np.reshape(data_batch, (-1, num_channels)) # 196N x 80
        # opq
        data_batch = opq.apply_py(np.ascontiguousarray(data_batch, dtype=np.float32))
        codes = pq.compute_codes(np.ascontiguousarray(data_batch, dtype=np.float32)) # 196N x 8
        codes = np.reshape(codes, (-1, spatial_feat_dim, spatial_feat_dim, num_codebooks)) # N x 14 x 14 x 8

        # put codes and labels into buffer (dictionary)
        for j in range(len(batch_labels)):
            ix = int(batch_item_ixs[j])
            latent_dict[ix] = [codes[j], batch_labels[j]]
            rehearsal_ixs.append(ix)
            class_id_to_item_ix_dict[int(batch_labels[j])].append(ix)
            counter.update()
    print("Completed in {} secs".format(time.time() - start_time))
    return pq, opq, latent_dict, rehearsal_ixs, class_id_to_item_ix_dict

### end ###
