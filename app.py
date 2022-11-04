# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import multiprocessing as mp

import numpy as np
from PIL import Image


try:
    import detectron2
except:
    import os
    os.system('pip install git+https://github.com/facebookresearch/detectron2.git')

from detectron2.config import get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from open_vocab_seg import add_ovseg_config
from open_vocab_seg.utils import VisualizationDemo

import gradio as gr

import gdown

ckpt_url = 'https://drive.google.com/uc?id=1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy'
output = './ovseg_swinbase_vitL14_ft_mpt.pth'
gdown.download(ckpt_url, output, quiet=False)

def setup_cfg(config_file):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg


def inference(class_names, input_img):
    mp.set_start_method("spawn", force=True)
    config_file = './ovseg_swinB_vitL_demo.yaml'
    cfg = setup_cfg(config_file)

    demo = VisualizationDemo(cfg)

    class_names = class_names.split(',')
    img = read_image(input_img, format="BGR")
    _, visualized_output = demo.run_on_image(img, class_names)

    return Image.fromarray(np.uint8(visualized_output.get_image())).convert('RGB')


examples = [['Oculus, Ukulele', './resources/demo_samples/sample_03.jpeg'],
            ['Saturn V, toys, blossom', './resources/demo_samples/sample_01.jpeg'],
            ['Golden gate, yacht', './resources/demo_samples/sample_02.jpeg'],]
output_labels = ['segmentation map']

title = 'OVSeg'

description = """
Gradio Demo for Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP. \n
OVSeg could perform open vocabulary segmentation, you may input more classes (seperate by comma). You may click on of the examples or upload your own image. \n
It might take some time to process. Cheers!
"""

article = """
<p style='text-align: center'>
<a href='https://arxiv.org/abs/2210.04150' target='_blank'>
Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP
</a>
|
<a href='https://github.com/facebookresearch/ov-seg' target='_blank'>Github Repo</a></p>
"""

gr.Interface(
    inference,
    inputs=[
        gr.inputs.Textbox(
            lines=1, placeholder=None, default='', label='class names'),
        gr.inputs.Image(type='filepath')
    ],
    outputs=gr.outputs.Image(label='segmentation map'),
    title=title,
    description=description,
    article=article,
    examples=examples).launch(enable_queue=True)
