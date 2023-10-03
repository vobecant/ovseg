# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import copy
import time

import clip
import cv2
import numpy as np
import open_clip
import torch
from PIL import Image
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures import BitMasks
from detectron2.utils.visualizer import ColorMode, Visualizer
from matplotlib import pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import SamPredictor
from torch.nn import functional as F

from open_vocab_seg.modeling.clip_adapter.adapter import PIXEL_MEAN, PIXEL_STD
from open_vocab_seg.modeling.clip_adapter.utils import crop_with_mask
from pca_visualization import pca_rgb_projection

VILD_PROMPT = [
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]


class PromptExtractor(torch.nn.Module):
    def __init__(self, templates=VILD_PROMPT):
        super().__init__()
        self.templates = templates

    def forward(self, noun_list, clip_model: torch.nn.Module):
        text_features_bucket = []
        for template in self.templates:
            noun_tokens = [clip.tokenize(template.format(noun)) for noun in noun_list]
            text_inputs = torch.cat(noun_tokens).to(
                clip_model.text_projection.data.device
            )
            text_features = clip_model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features_bucket.append(text_features)
        del text_inputs
        # ensemble by averaging
        text_features = torch.stack(text_features_bucket).mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features


def get_text_features(clip_model, class_names, prompt_creator):
    text_features = prompt_creator(
        class_names, clip_model
    )
    return text_features


def matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS for multi-class masks.
    Args:
        seg_masks (Tensor): shape (n, h, w)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss'
        sigma (float): std in gaussian method
        sum_masks (Tensor): The sum of seg_masks
    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).float()
    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # iou.
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update


class OVSegPredictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, original_image, class_names):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width, "class_names": class_names}
            predictions, dense_features = self.model([inputs])
            predictions = predictions[0]
            return predictions, dense_features


class OVSegVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE, class_names=None):
        super().__init__(img_rgb, metadata, scale, instance_mode)
        self.class_names = class_names

    def draw_sem_seg(self, sem_seg, area_threshold=None, alpha=0.8):
        """
        Draw semantic segmentation predictions/labels.

        Args:
            sem_seg (Tensor or ndarray): the segmentation of shape (H, W).
                Each value is the integer label of the pixel.
            area_threshold (int): segments with less than `area_threshold` are not drawn.
            alpha (float): the larger it is, the more opaque the segmentations are.

        Returns:
            output (VisImage): image object with visualizations.
        """
        if isinstance(sem_seg, torch.Tensor):
            sem_seg = sem_seg.numpy()
        labels, areas = np.unique(sem_seg, return_counts=True)
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]
        class_names = self.class_names if self.class_names is not None else self.metadata.stuff_classes

        for label in filter(lambda l: l < len(class_names), labels):
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[label]]
            except (AttributeError, IndexError):
                mask_color = None

            binary_mask = (sem_seg == label).astype(np.uint8)
            text = class_names[label]
            self.draw_binary_mask(
                binary_mask,
                color=mask_color,
                edge_color=(1.0, 1.0, 240.0 / 255),
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )
        return self.output

    def draw_rgb(self, rgb, area_threshold=None, alpha=0.8):
        """
        Draw semantic segmentation predictions/labels.

        Args:
            rgb (Tensor or ndarray): the RGB image of shape (H, W, 3).
            area_threshold (int): segments with less than `area_threshold` are not drawn.
            alpha (float): the larger it is, the more opaque the segmentations are.

        Returns:
            output (VisImage): image object with visualizations.
        """
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.numpy()
        labels, areas = np.unique(rgb, return_counts=True)
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]
        class_names = self.class_names if self.class_names is not None else self.metadata.stuff_classes

        for label in filter(lambda l: l < len(class_names), labels):
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[label]]
            except (AttributeError, IndexError):
                mask_color = None

            binary_mask = (rgb == label).astype(np.uint8)
            text = class_names[label]
            self.draw_binary_mask(
                binary_mask,
                color=mask_color,
                edge_color=(1.0, 1.0, 240.0 / 255),
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )
        return self.output


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            raise NotImplementedError
        else:
            self.predictor = OVSegPredictor(cfg)

    def run_on_image(self, image, class_names, compute_pca=False):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = OVSegVisualizer(image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)
        visualizer_dense = OVSegVisualizer(copy.deepcopy(image), self.metadata, instance_mode=self.instance_mode,
                                           class_names=class_names)
        visualizer_seg = OVSegVisualizer(copy.deepcopy(image), self.metadata, instance_mode=self.instance_mode,
                                         class_names=class_names)

        image_rgb = copy.deepcopy(image)
        time_s = time.time()
        with torch.no_grad():
            predictions, dense_features = self.predictor(image, class_names)

        if "sem_seg" in predictions:
            r = predictions["sem_seg"]
            blank_area = (r[0] == 0)
            pred_mask = r.argmax(dim=0).to('cpu')
            pred_mask[blank_area] = 255
            pred_mask = np.array(pred_mask, dtype=np.int)

            vis_output = visualizer.draw_sem_seg(
                pred_mask
            )
        else:
            raise NotImplementedError

        if compute_pca:
            _npy_output_pca = None
        else:
            _npy_output_pca = None

        visualized_output_dense = None

        return predictions, vis_output, visualized_output_dense, _npy_output_pca, None, dense_features


class SAMVisualizationDemo(object):
    def __init__(self, cfg, granularity, sam_path, ovsegclip_path, instance_mode=ColorMode.IMAGE, parallel=False,
                 crop_nms_thresholds=None, crop_overlap_ratios=None, slic_params=None, pred_iou_thresh=0.88,
                 points_given=False):
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.slic_params = slic_params

        self.parallel = parallel
        self.granularity = granularity

        self.points_given = points_given

        if self.slic_params is None:
            sam = sam_model_registry["vit_l"](checkpoint=sam_path).cuda()
            '''
            Hyperparameters to experiment with:
            - crop_nms_thresh: If we set it lower, it might filter out fewer masks.
            - crop_overlap_ratio: Sets the degree to which crops overlap. In [0, 1]. The higher, the higher overlap.
            '''
            if not points_given:
                self.predictor = SamAutomaticMaskGenerator(sam,
                                                           # points_per_batch=16,
                                                           points_per_batch=32,
                                                           # points_per_side=16,
                                                           points_per_side=32,
                                                           pred_iou_thresh=pred_iou_thresh
                                                           )
            else:
                self.predictor = SamPredictor(sam)
        self.crop_nms_thresholds = crop_nms_thresholds if crop_nms_thresholds is not None else [
            self.predictor.crop_nms_thresh]
        self.crop_overlap_ratios = crop_overlap_ratios if crop_overlap_ratios is not None else [
            self.predictor.crop_overlap_ratio]

        if not isinstance(self.crop_nms_thresholds, list):
            self.crop_nms_thresholds = [self.crop_nms_thresholds]
        if not isinstance(self.crop_nms_thresholds, list):
            self.crop_overlap_ratios = [self.crop_overlap_ratios]

        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=ovsegclip_path)
        self.clip_model.cuda()

        self.prompt_creator = PromptExtractor()

    def run_on_image(self, i_iter, image, class_names, compute_pca, show, features_only, verbose=False,
                     projections=None, debug=False, multimask_output=True, nms_thr=0.5, image_path=None,
                     **kwargs):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if show:
            visualizer = OVSegVisualizer(image, self.metadata, instance_mode=self.instance_mode,
                                         class_names=class_names)
            visualizer_dense = OVSegVisualizer(copy.deepcopy(image), self.metadata, instance_mode=self.instance_mode,
                                               class_names=class_names)
            visualizer_seg = OVSegVisualizer(copy.deepcopy(image), self.metadata, instance_mode=self.instance_mode,
                                             class_names=class_names)
            image_rgb = copy.deepcopy(image)
        time_s = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast():
            if self.points_given:
                self.predictor.set_image(image)
                pred_masks, scores = [], []
                projections = projections[:, None, [1, 0]]
                n_queries = projections.shape[0]
                for point in projections:
                    masks_cur, confs, _ = self.predictor.predict(point_coords=point, point_labels=np.array([1]),
                                                                 multimask_output=multimask_output
                                                                 )
                    pred_masks.extend(masks_cur)
                    scores.extend(confs)

                pred_masks = [mask[None, :, :] for mask in pred_masks]
                scores = np.array(scores)
                del masks_cur, _
            else:
                masks = self.predict_sam_multiparam(image)
                pred_masks = [masks[i]['segmentation'][None, :, :] for i in range(len(masks))]
                del masks
                n_queries = self.predictor.points_per_side ** 2
            # masks = self.predictor.generate(image)

        # for pred_mask in pred_masks:
        #     plt.imshow(image_rgb)
        #     plt.imshow(pred_mask[0], cmap='gray', alpha=0.5)
        #     plt.show()

        pred_masks = np.row_stack(pred_masks)
        sum_masks = torch.from_numpy(pred_masks.sum((1, 2)).astype(float)).cuda()
        try:
            res = matrix_nms(torch.from_numpy(pred_masks).cuda(), torch.zeros(pred_masks.shape[0]).cuda(),
                             torch.from_numpy(scores).cuda(), sum_masks=sum_masks)
        except:
            n_best = 200
            print(f'[Iter {i_iter}] Retrieving from OOM! Had {len(scores)} inputs to NMS, #queries: {n_queries}.')
            if image_path is not None:
                print(f'[Iter {i_iter}] Input image path: {image_path}')
            print(f'[Iter {i_iter}] Limiting masks to just one mask per query and '
                  f'take {n_best} with the largets scores!')
            torch.cuda.empty_cache()

            # limit to one output per query
            pred_masks = pred_masks[::3]
            scores = scores[::3]
            sum_masks = sum_masks[::3]

            sorted_scores_idx = (-scores).argsort()[:n_best]
            pred_masks = pred_masks[sorted_scores_idx]
            scores = scores[sorted_scores_idx]
            sum_masks = sum_masks[sorted_scores_idx].cpu()
            res = matrix_nms(torch.from_numpy(pred_masks), torch.zeros(pred_masks.shape[0]),
                             torch.from_numpy(scores), sum_masks=sum_masks)

        del sum_masks
        n_masks = 100
        picked = (-res).argsort()[:n_masks].cpu()
        n_picked = len(picked)
        pred_masks = pred_masks[picked]
        scores = scores[picked]

        time_s_extra = time.time()
        occupied_mask = pred_masks.sum(0) > 0
        uncovered_points_idx = np.where(~occupied_mask[projections[..., 1], projections[..., 0]])
        uncovered_points = projections[uncovered_points_idx][:, None, :]
        if len(uncovered_points) > 0:
            extra_masks, extra_scores = [], []
            for point in uncovered_points:
                masks_cur, confs, _ = self.predictor.predict(point_coords=point, point_labels=np.array([1]),
                                                             multimask_output=multimask_output
                                                             )
                extra_masks.extend(masks_cur)
                extra_scores.extend(confs)
            extra_masks = np.stack(extra_masks)
            extra_scores = np.asarray(extra_scores)
            pred_masks = np.concatenate((extra_masks, pred_masks))
            scores = np.concatenate((extra_scores, scores))
        time_elapsed_extra = time.time() - time_s_extra

        time_elapsed = time.time() - time_s
        if verbose:
            print('SAM-related time: {:.1f}s (NMS+extra {:.1f}s), {} masks ({} after NMS), multimask_output={}, '
                  '# queries: {}'.format(time_elapsed, time_elapsed_extra, pred_masks.shape[0], n_picked,
                                         multimask_output, n_queries))

        if debug:
            plt.imshow(image)
            plt.imshow(pred_masks.sum(0) == 0, cmap='gray', alpha=0.5)
            plt.scatter(projections[:, 0, 0], projections[:, 0, 1], s=10, c='r')
            plt.show()

        segmentation_output = visualizer_seg.draw_sem_seg(pred_masks.argmax(0)) if show else None
        pred_masks = BitMasks(pred_masks)
        bboxes = pred_masks.get_bounding_boxes()

        mask_fill = [255.0 * c for c in PIXEL_MEAN]

        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        regions = []
        kept_pred_masks = []
        for bbox, mask in zip(bboxes, pred_masks):
            l, t, r, b = bbox
            w, h = int(abs(r - l)), int(abs(b - t))
            if w == 0 or h == 0:
                kept_pred_masks.append(False)
                continue
            kept_pred_masks.append(True)
            region, _ = crop_with_mask(
                image,
                mask,
                bbox,
                fill=mask_fill,
            )
            regions.append(region.unsqueeze(0))
        pred_masks = pred_masks[kept_pred_masks]
        regions = [F.interpolate(r.to(torch.float), size=(224, 224), mode="bicubic") for r in regions]

        time_s = time.time()
        pixel_mean = torch.tensor(PIXEL_MEAN).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(PIXEL_STD).reshape(1, -1, 1, 1)
        imgs = [(r / 255.0 - pixel_mean) / pixel_std for r in regions]
        del regions
        imgs = torch.cat(imgs)
        if len(class_names) == 1:
            class_names.append('others')
        txts = [f'a photo of {cls_name}' for cls_name in class_names]
        text = open_clip.tokenize(txts)

        img_batches = torch.split(imgs, 32, dim=0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            # text_features = self.clip_model.encode_text(text.cuda())
            # text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = get_text_features(self.clip_model, class_names, self.prompt_creator)
            image_features = []
            for img_batch in img_batches:
                image_feat = self.clip_model.encode_image(img_batch.cuda().half())
                image_feat /= image_feat.norm(dim=-1, keepdim=True)
                image_features.append(image_feat.detach())
            image_features = torch.cat(image_features, dim=0)
            class_preds = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        select_cls = torch.zeros_like(class_preds)
        time_elapsed_features = time.time() - time_s
        if verbose: print('Feature extraction time: {:.1f}s'.format(time_elapsed_features))

        semseg = None
        if not features_only:
            max_scores, select_mask = torch.max(class_preds, dim=0)
            if len(class_names) == 2 and class_names[-1] == 'others':
                select_mask = select_mask[:-1]
            if self.granularity < 1:
                thr_scores = max_scores * self.granularity
                select_mask = []
                if len(class_names) == 2 and class_names[-1] == 'others':
                    thr_scores = thr_scores[:-1]
                for i, thr in enumerate(thr_scores):
                    cls_pred = class_preds[:, i]
                    locs = torch.where(cls_pred > thr)
                    select_mask.extend(locs[0].tolist())
            for idx in select_mask:
                select_cls[idx] = class_preds[idx]
            semseg = torch.einsum("qc,qhw->chw", select_cls.float(), pred_masks.tensor.float().cuda())

        # pred_masks: shape NxHxW (num. masks, height, width)
        # image_features: shape NxC (num. masks, number of feature channels)
        dense_features = torch.einsum("qc,qhw->chw", image_features.float(), pred_masks.tensor.float().cuda())
        dense_features /= pred_masks.tensor.float().cuda().sum(0).clip(min=1.)

        semseg_dense = torch.einsum("qc,chw->qhw", text_features.float(), dense_features.float().cuda()).argmax(0).cpu()

        vis_output = pred_mask = None
        if show:
            r = semseg
            blank_area = (r[0] == 0)
            pred_mask = r.argmax(dim=0).to('cpu')
            pred_mask[blank_area] = 255
            pred_mask = np.array(pred_mask, dtype=np.int)
            vis_output = visualizer.draw_sem_seg(
                pred_mask
            )

        # r = semseg_dense
        # blank_area = (r[0] == 0)
        # pred_mask_dense = r.argmax(dim=0).to('cpu')
        # pred_mask_dense[blank_area] = 255
        # pred_mask_dense = np.array(pred_mask_dense, dtype=np.int)
        pred_mask_dense = semseg_dense.cpu().numpy()

        # PCA RGB visualization
        if compute_pca:
            time_s = time.time()
            pred_mask_pca, *_ = pca_rgb_projection(dense_features)
            image_rgb_pil = Image.fromarray(image_rgb)
            pred_mask_pca_pil = Image.fromarray(pred_mask_pca)
            time_elapsed = time.time() - time_s
            print('PCA projection to RGB computed in {:.1f}s'.format(time_elapsed))

            image_rgb_pil = image_rgb_pil.convert("RGBA")
            pred_mask_pca_pil = pred_mask_pca_pil.convert("RGBA")

            new_img = np.array(Image.blend(image_rgb_pil, pred_mask_pca_pil, 0.8))
        else:
            new_img = None

        vis_output_dense = None
        if show:
            vis_output_dense = visualizer_dense.draw_sem_seg(
                pred_mask_dense
            )

        del semseg_dense, pred_mask_dense, semseg, pred_mask, pred_masks, image_features

        return None, vis_output, vis_output_dense, new_img, segmentation_output, dense_features

    def predict_sam_multiparam(self, image):
        '''
        Hyperparameters to experiment with:
        - crop_nms_thresh: If we set it lower, it might filter out fewer masks.
        - crop_overlap_ratio: Sets the degree to which crops overlap. In [0, 1]. The higher, the higher overlap.
        '''
        masks_all = []
        for crop_nms_thresh in self.crop_nms_thresholds:
            self.predictor.crop_nms_thresh = crop_nms_thresh
            for crop_overlap_ratio in self.crop_overlap_ratios:
                self.predictor.crop_overlap_ratio = crop_overlap_ratio
                masks_cur = self.predictor.generate(image)
                masks_all.extend(masks_cur)
        return masks_all
