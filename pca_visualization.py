import numpy as np
import cv2
import torch
from sklearn.decomposition import PCA


def pca_rgb_projection(features, standardize=False, mean=None, std=None, pca=None, reuse_pca=None):
    # features are of shape [N,dim]
    # X = features.transpose(1, 2, 0)
    # print(f'features.shape: {features.shape}')
    ndim = len(features.shape)
    if ndim == 4:
        # shape [B, C, H, W]
        features = features[0]
        C, H, W = features.shape
        X = features.view(C, H * W).T
    elif ndim == 3:
        C, H, W = features.shape
        X = features.view(C, H * W).T
    else:
        X = features
        H, W = None, None

    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    # Reshape to have shape (nb_pixel x dim)
    Xt_all = X
    Xt = Xt_all

    # print(f'Xt.shape: {Xt.shape}')

    # Normalize
    if standardize:
        if mean is None:
            mean = np.mean(Xt, axis=-1)[:, np.newaxis]
        if std is None:
            std = np.std(Xt, axis=-1)[:, np.newaxis]
        Xtn = (Xt - mean) / std
        Xtn_all = Xtn
    else:
        mean = std = None
        Xtn = Xt
        Xtn_all = Xt_all

    # Xtn = np.nan_to_num(Xtn)
    # Xtn_all = np.nan_to_num(Xtn_all)

    nb_comp = 3
    if pca is None:
        # Apply PCA
        pca = PCA(n_components=nb_comp)
        pca.fit(Xtn)
    else:
        # print('Using precomputed PCA.')
        pass
    projected = pca.fit_transform(Xtn_all)
    projected = projected.reshape(X.shape[0], nb_comp)

    # normalizing between 0 to 255
    PC_n = np.zeros((X.shape[0], nb_comp))
    for i in range(nb_comp):
        PC_n[:, i] = cv2.normalize(projected[:, i],
                                   np.zeros((X.shape[0])), 0, 255, cv2.NORM_MINMAX).squeeze().clip(0, 256)
    PC_n = PC_n.astype(np.uint8)

    if H is not None and W is not None:
        PC_n.resize(H, W, 3)

    if not reuse_pca:
        mean = std = pca = None
    return PC_n, (mean, std, pca)