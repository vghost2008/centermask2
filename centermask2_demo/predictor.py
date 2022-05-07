# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor,SimplePredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from centermask.config import get_cfg
import os.path as osp
import torch.nn as nn

pdir_path = osp.dirname(osp.dirname(__file__))
cfg_file = osp.join(pdir_path,"configs/centermask/centermask_V_39_eSE_FPN_ms_3x.yaml")


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.SEGMENTATION):
        #ColorMode.IMAGE
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

        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        image = image[...,::-1]
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "inst" in predictions:
            visualizer.vis_inst(predictions["inst"])
        if "bases" in predictions:
            self.vis_bases(predictions["bases"])
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device))
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def vis_bases(self, bases):
        basis_colors = [[2, 200, 255], [107, 220, 255], [30, 200, 255], [60, 220, 255]]
        bases = bases[0].squeeze()
        bases = (bases / 8).tanh().cpu().numpy()
        num_bases = len(bases)
        fig, axes = plt.subplots(nrows=num_bases // 2, ncols=2)
        for i, basis in enumerate(bases):
            basis = (basis + 1) / 2
            basis = basis / basis.max()
            basis_viz = np.zeros((basis.shape[0], basis.shape[1], 3), dtype=np.uint8)
            basis_viz[:, :, 0] = basis_colors[i][0]
            basis_viz[:, :, 1] = basis_colors[i][1]
            basis_viz[:, :, 2] = np.uint8(basis * 255)
            basis_viz = cv2.cvtColor(basis_viz, cv2.COLOR_HSV2RGB)
            axes[i // 2][i % 2].imshow(basis_viz)
        plt.show()

class InferenceModel(nn.Module):
    def __init__(self, cfg=None, instance_mode=ColorMode.SEGMENTATION):
        #ColorMode.IMAGE
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        super().__init__()
        if cfg is None:
            cfg = self.setup_cfg()
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor = SimplePredictor(cfg)
        self.add_module("model",self.predictor.model)

    def setup_cfg(self):
        cfg = get_cfg()
        cfg.merge_from_file(cfg_file)
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.4
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
        cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.4
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.4
        cfg.MODEL.WEIGHTS = osp.join(pdir_path,cfg.MODEL.WEIGHTS)
        cfg.freeze()
        return cfg

    def forward(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        with torch.no_grad():
            predictions = self.predictor(image)
        return predictions
