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

CLASS_MAP = {
    1: 0,
    5: 0,
    7: 0,
    8: 0,
    10: 0,
    11: 0,
    13: 0,
    19: 0,
    20: 0,
    0: 0,
    29: 0,
    31: 0,
    9: 1,
    14: 2,
    15: 3,
    16: 3,
    17: 4,
    18: 5,
    21: 6,
    2: 7,
    3: 7,
    4: 7,
    6: 7,
    12: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    30: 16
}

NAMES2CLS_EXTENDED = [
    0, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 9, 9, 10, 10, 10, 11, 11, 11, 11, 11, 12, 13, 14, 15,
    16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 19, 19, 20, 20, 20, 20, 21, 21, 21, 22, 22, 22, 22,
    23, 23, 23, 23, 23, 24, 24, 25, 25, 25, 25, 25, 25, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28,
    28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 29, 30, 30, 30, 30, 30, 30, 30, 31
]

NAMES2CLS_PREDICTED = [CLASS_MAP[cls] for cls in NAMES2CLS_EXTENDED]

CLASS_NAMES = [
    [
        'Any lidar return that does not correspond to a physical object, such as dust, vapor, noise, fog, raindrops, smoke and reflections.'],
    ['animal', 'cat', 'dog', 'rat', 'deer', 'bird'], ['Adult.'], ['Child.'], ['Construction worker'],
    ['skateboard', 'segway'], ['Police officer.'], ['Stroller'], ['Wheelchair'],
    ['Temporary road barrier to redirect traffic.', 'concrete barrier', 'metal barrier', 'water barrier'],
    ['Movable object that is left on the driveable surface.', 'tree branch', 'full trash bag'],
    ['Object that a pedestrian may push or pull.', 'dolley', 'wheel barrow', 'garbage-bin', 'shopping cart'],
    ['traffic cone.'], ['Area or device intended to park or secure the bicycles in a row.'], ['Bicycle'],
    ['Bendy bus'], ['Rigid bus'],
    ['Vehicle designed primarily for personal use.', 'car', 'vehicle', 'sedan', 'hatch-back', 'wagon', 'van',
     'mini-van', 'SUV', 'jeep'], ['Vehicle designed for construction.', 'crane'],
    ['ambulance', 'ambulance vehicle'],
    ['police vehicle', 'police car', 'police bicycle', 'police motorcycle'],
    ['motorcycle', 'vespa', 'scooter'], ['trailer', 'truck trailer', 'car trailer', 'bike trailer'],
    ['Vehicle primarily designed to haul cargo.', 'pick-up', 'lorry', 'truck', 'semi-tractor'],
    ['Paved surface that a car can drive.', 'Unpaved surface that a car can drive.'],
    ['traffic island', 'delimiter', 'rail track', 'stairs', 'lake', 'river'],
    ['sidewalk', 'pedestrian walkway', 'bike path'], ['grass', 'rolling hill', 'soil', 'sand', 'gravel'],
    [
        'man-made structure', 'building', 'wall', 'guard rail', 'fence', 'pole', 'drainage', 'hydrant', 'flag',
        'banner', 'street sign', 'electric circuit box', 'traffic light', 'parking meter', 'stairs'
    ],
    [
        'Points in the background that are not distinguishable, or objects that do not match any of the above labels.'
    ],
    ['bushes', 'bush', 'plants', 'plant', 'potted plant', 'tree', 'trees'],
    [
        'The vehicle on which the cameras, radar and lidar are mounted, that is sometimes visible at the bottom of the image.'
    ]
]

CLASS_NAMES_FLATTEN = [item for sublist in CLASS_NAMES for item in sublist]

assert len(CLASS_NAMES_FLATTEN) == len(NAMES2CLS_PREDICTED)

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


def prediction2cls(prediction):
    prediction_cls = NAMES2CLS_PREDICTED[prediction]
    return prediction_cls


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
        save_path = os.path.join(save_dir, spec, cam_name, img_name + '.pth')
        d = os.path.split(save_path)[0]
        if not os.path.exists(d):
            os.makedirs(d)
        if not debug and os.path.exists(save_path):
            continue

        projections = projections_unq = sam_output = None
        if projections_dir is not None:
            cam_name, img_name = input_img.split(os.path.sep)[-2:]
            img_name = img_name.split('.')[0]
            projections_path_unq = os.path.join(projections_dir, cam_name, f'{img_name}__pixels_vox_unq.npy')
            projections_unq = np.load(projections_path_unq)
            projections_path = os.path.join(projections_dir, cam_name, f'{img_name}__pixels.npy')
            projections = np.load(projections_path)

        img = read_image(input_img, format="BGR")
        _npy_output_pca = visualized_output_dense = None
        with torch.no_grad():

            if proposal_gen == 'MaskFormer':
                predictions, visualized_output, _, _, _, dense_features = demo.run_on_image(img, class_names,
                                                                                            show=args.show)
                sem_seg = predictions['sem_seg'].argmax(0)
            else:
                _, visualized_output, visualized_output_dense, _npy_output_pca, sam_output, dense_features = demo.run_on_image(
                    i_iter=i, image=img, class_names=class_names, compute_pca=compute_pca, show=show,
                    features_only=features_only,
                    verbose=verbose, projections=projections_unq, debug=debug, multimask_output=multimask_output,
                    image_path=input_img)

        if _npy_output_pca is not None:
            plt.imshow(_npy_output_pca)
            plt.title('PCA visualization')
            plt.show()

        if args.show:
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
            rows, cols = projections.T
            predictions_projected = sem_seg[rows, cols]
            torch.save(predictions_projected, save_path)

        del dense_features


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
    parser.add_argument('--proposal-gen', type=str, default='MaskFormer')  # Segment_Anything
    args = parser.parse_args()

    save_dir = args.save_dir
    assert save_dir is not None, "You need to set --save-dir!!!"
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start = end = None
    if args.start_end is not None:
        start, end = args.start_end

    images_file = '/nfs/datasets/nuscenes/val_image_paths.txt'
    nusc_root = '/nfs/datasets/nuscenes'
    with open(images_file, 'r') as f:
        image_paths = [l.strip() for l in f.readlines()]
        image_paths = [os.path.join(nusc_root, p) for p in image_paths]

    if args.start_end is not None:
        start, end = args.start_end
        print(f'Take paths with indices {start}-{end} out of {len(image_paths)} total images.')
        image_paths = image_paths[start:end]

    inference(CLASS_NAMES_FLATTEN,
              args.proposal_gen,
              GRANULARITY, image_paths, compute_pca=args.compute_pca, show=args.show, features_only=args.features_only,
              verbose=args.verbose, projections_dir=args.projections_path, save_dir=save_dir, debug=args.debug,
              points_given=args.points_given, multimask_output=not args.no_multimask)
