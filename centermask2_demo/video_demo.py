import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))
from demo_toolkit import *
from boedemo.predictor import VisualizationDemo
from centermask.config import get_cfg

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

pdir_path = osp.dirname(osp.dirname(__file__))
cfg_file = osp.join(pdir_path,"configs/centermask/centermask_V_39_eSE_FPN_ms_3x.yaml")

class Model:
    def __init__(self):
        cfg = self.setup_cfg()
        self.model = VisualizationDemo(cfg)

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

    def __call__(self, img):
        results,vis_img= self.model.run_on_image(img)
        img = vis_img.get_image()
        return img

if __name__ == "__main__":
    vd = VideoDemo(Model(),save_path="tmp1.mp4",show_video=False,max_frame_cn=None)
    vd.preprocess = lambda x:resize_short_size(x,640)
    video_path = None
    if len(sys.argv)>1:
        video_path = sys.argv[1]
    vd.inference_loop(video_path)
    vd.close()
