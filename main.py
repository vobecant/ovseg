# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import argparse
import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np
import torch.cuda
from tqdm import tqdm

try:
    import detectron2
except:
    import os

    os.system('pip install git+https://github.com/facebookresearch/detectron2.git')

from detectron2.config import get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from open_vocab_seg import add_ovseg_config
from open_vocab_seg.utils import VisualizationDemo, SAMVisualizationDemo

NUSC_NAMES = [
    'adult pedestrian', 'child pedestrian', 'wheelchair', 'stroller', 'personal mobility', 'police officer',
    'construction worker', 'animal', 'car', 'motorcycle', 'bicycle', 'bendy bus', 'rigid bus', 'truck',
    'construction vehicle', 'ambulance vehicle', 'police car', 'trailer', 'barrier', 'traffic cone', 'debris',
    'bicycle rack', 'driveable surface', 'sidewalk', 'terrain', 'other flat', 'manmade', 'vegetation', 'sky'
]

NUSC_NAMES_SHORT = [
    # 'noise',
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
    'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
]

# barrier,bicycle,bus,car,construction_vehicle,motorcycle,pedestrian,traffic_cone,trailer,truck,driveable_surface,other_flat,sidewalk,terrain,manmade,vegetation

GRANULARITY = 0.9

CROP_NMS_THRESH: float = [0.3]  # [0.7, 0.5]
CROP_OVERLAP_RATIO: float = [0.5]  # [512 / 1500, 0.5]
PRED_IOU_THRESH = 0.5  # 0.88


def setup_cfg(config_file):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg


def inference(class_names, proposal_gen, granularity, input_img_paths, compute_pca=False, show=False,
              features_only=False, verbose=False, projections_dir=None, save_dir=None, debug=False, points_given=False,
              multimask_output=True):
    mp.set_start_method("spawn", force=True)
    config_file = './ovseg_swinB_vitL_demo.yaml'
    cfg = setup_cfg(config_file)
    if proposal_gen == 'MaskFormer':
        demo = VisualizationDemo(cfg)
        spec = 'maskformer'
    elif proposal_gen.lower() in ['segment_anything', 'slic']:
        slic_params = None
        if proposal_gen.lower() == 'slic':
            slic_params = ...
        demo = SAMVisualizationDemo(cfg, granularity, './sam_vit_l_0b3195.pth', './ovseg_clip_l_9a1909.pth',
                                    crop_overlap_ratios=CROP_OVERLAP_RATIO, crop_nms_thresholds=CROP_NMS_THRESH,
                                    slic_params=slic_params, pred_iou_thresh=PRED_IOU_THRESH, points_given=points_given)
        crop_overlaps = '+'.join([str(co) for co in CROP_OVERLAP_RATIO])
        crop_nms_thrs = '+'.join([str(cn) for cn in CROP_NMS_THRESH])

        spec = f'sam-{granularity}g-{crop_overlaps}co-{crop_nms_thrs}cn'
    class_names = class_names.split(',') if not type(class_names) == list else class_names

    if not isinstance(input_img_paths, list):
        input_img_paths = [input_img_paths]

    for i, input_img in enumerate(tqdm(input_img_paths)):
        # if i == 3: break

        cam_name, img_name = input_img.split(os.path.sep)[-2:]
        img_name = img_name.split('.')[0]
        save_path = os.path.join(save_dir, spec, cam_name, img_name.replace('.jpg', '.pth'))
        d = os.path.split(save_path)[0]
        if not os.path.exists(d):
            os.makedirs(d)
        if not debug and os.path.exists(save_path):
            continue

        projections = None
        if projections_dir is not None:
            cam_name, img_name = input_img.split(os.path.sep)[-2:]
            img_name = img_name.split('.')[0]
            projections_path_unq = os.path.join(projections_dir, cam_name, f'{img_name}__pixels_vox_unq.npy')
            projections_unq = np.load(projections_path_unq)
            projections_path = os.path.join(projections_dir, cam_name, f'{img_name}__pixels.npy')
            projections = np.load(projections_path)

        img = read_image(input_img, format="BGR")
        with torch.no_grad():
            _, visualized_output, visualized_output_dense, _npy_output_pca, sam_output, dense_features = demo.run_on_image(
                i, img, class_names, compute_pca=compute_pca, show=show, features_only=features_only,
                verbose=verbose, projections=projections_unq, debug=debug, multimask_output=multimask_output,
                image_path=input_img)

        if _npy_output_pca is not None:
            plt.imshow(_npy_output_pca)
            plt.title('PCA visualization')
            plt.show()

        if visualized_output is not None:
            plt.imshow(np.uint8(visualized_output.get_image()))
            plt.title('OV-Seg output')
            plt.show()

        if visualized_output_dense is not None:
            plt.imshow(np.uint8(visualized_output_dense.get_image()))
            plt.title('OV-Seg output, dense')
            plt.show()

        if sam_output is not None:
            plt.imshow(np.uint8(sam_output.get_image()))
            plt.title('SAM output')
            plt.show()

        if save_dir is not None:
            if projections is not None:
                rows, cols = projections.T
                dense_features = dense_features[:, rows, cols]
            torch.save(dense_features, save_path)

        del dense_features
        # return Image.fromarray(np.uint8(visualized_output.get_image())).convert('RGB')

        # torch.cuda.empty_cache()


examples = [['Saturn V, toys, desk, wall, sunflowers, white roses, chrysanthemums, carnations, green dianthus',
             'Segment_Anything', 0.8, './resources/demo_samples/sample_01.jpeg'],
            [
                'red bench, yellow bench, blue bench, brown bench, green bench, blue chair, yellow chair, green chair, brown chair, yellow square painting, barrel, buddha statue',
                'Segment_Anything', 0.8, './resources/demo_samples/sample_04.png'],
            ['pillow, pipe, sweater, shirt, jeans jacket, shoes, cabinet, handbag, photo frame', 'Segment_Anything',
             0.8, './resources/demo_samples/sample_05.png'],
            ['Saturn V, toys, blossom', 'MaskFormer', 1.0, './resources/demo_samples/sample_01.jpeg'],
            ['Oculus, Ukulele', 'MaskFormer', 1.0, './resources/demo_samples/sample_03.jpeg'],
            ['Golden gate, yacht', 'MaskFormer', 1.0, './resources/demo_samples/sample_02.jpeg'], ]
output_labels = ['segmentation map']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--compute-pca', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--features-only', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--points-given', action='store_true')
    parser.add_argument('--no-multimask', action='store_true')
    parser.add_argument('--projections-path',
                        # default="/nfs/datasets/nuscenes/features/projections",
                        default=None,
                        type=str)
    parser.add_argument('--save-dir', default=None, type=str)
    parser.add_argument('--start-end', type=int, nargs='+', default=None)
    parser.add_argument('=-proposal-gen', type=str, default='Segment_Anything')  # 'MaskFormer
    args = parser.parse_args()

    save_dir = args.save_dir
    assert save_dir is not None, "You need to set --save-dir!!!"
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images_file = '/nfs/datasets/nuscenes/nerf_data/scene-0061_paths.txt'
    with open(images_file, 'r') as f:
        image_paths = f.readline().split(',')

    images_file = '/nfs/datasets/nuscenes/paths_mini.txt'
    nusc_root = '/nfs/datasets/nuscenes'
    with open(images_file, 'r') as f:
        image_paths = [l.strip() for l in f.readlines()]
        image_paths = [os.path.join(nusc_root, p) for p in image_paths]

    missing = [
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984239412460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984239420339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984239404844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984239437525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984239447423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984241412460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984241420339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984241404844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984241437525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984241447423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984241427893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984241912460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984241920339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984241904844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984241937525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984241947423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984241927893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984242412460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984242420339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984242404844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984242437525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984242447423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984242427893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984242912460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984242920339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984242904844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984242937525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984242947423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984242927893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984243412460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984243420339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984243404844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984243437525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984243447423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984243427893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984243912460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984243920339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984243904844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984243937525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984243947423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984243927893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984244412460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984244420339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984244404844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984244437534',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984244447423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984244427893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984244912460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984244920347',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984244904844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984244937525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984244947423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984244927893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984245412460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984245420339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984245404844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984245437525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984245447423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984245427893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984245912460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984245920339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984245904844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984245937525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984245947423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984245927893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984246412460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984246420339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984246404844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984246437525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984246447423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984246427893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984246912460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984246920339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984246904844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984246937525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984246947423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984246927893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984247412460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984247420339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984247404844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984247437525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984247447423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984247427893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984247912460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984247920339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984247904844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984247937525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984247947423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984247927893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984248412460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984248420339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984248404844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984248437525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984248447423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984248427893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984248912460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984248920339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984248904844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984248937525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984248947423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984248927893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984249412460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984249420339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984249404844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984249437525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984249447423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984249427893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984249912460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984249920351',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984249904844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984249937525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984249947423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984249927893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984250412460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984250420339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984250404844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984250437525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984250447423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984250427893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984250912460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984250920339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984250904844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984250937525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984250947423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984250927893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984251412460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984251420339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984251404844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984251437525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984251447423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984251427893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984251912460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984251920339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984251904844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984251937525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984251947423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984251927893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984252412460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984252420339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984252404844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984252437525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984252447423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984252427893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984252912460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984252920339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984252904844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984252937527',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984252947423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984252927893',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984253412460',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984253420339',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_FRONT_LEFT/n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984253404844',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK/n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984253437525',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_LEFT/n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984253447423',
        '/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn/CAM_BACK_RIGHT/n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984253427893']
    image_paths = [path.replace('/nfs/datasets/nuscenes/ovseg_features_projections_all/sam-0.9g-0.5co-0.3cn',
                                os.path.join(nusc_root, 'samples')) + '.jpg'
                   for path in missing]

    if args.start_end is not None:
        start, end = args.start_end
        print(f'Take paths with indices {start}-{end} out of {len(image_paths)} total images.')
        image_paths = image_paths[start:end]

    inference(NUSC_NAMES_SHORT,
              args.proposal_gen,
              # 'Segment_Anything',
              # 'MaskFormer',
              GRANULARITY, image_paths, compute_pca=args.compute_pca, show=args.show, features_only=args.features_only,
              verbose=args.verbose, projections_dir=args.projections_path, save_dir=save_dir, debug=args.debug,
              points_given=args.points_given, multimask_output=not args.no_multimask)
