import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))
from demo_toolkit import *
from centermask2_demo.predictor import VisualizationDemo
from centermask.config import get_cfg
import wml_utils as wmlu
import img_utils as wmli

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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
        img = wmli.resize_short_size(img,640)
        results,vis_img= self.model.run_on_image(img)
        img = vis_img.get_image()
        return img

if __name__ == "__main__":
    file_dir = "/home/wj/ai/mldata1/0day/wear"
    save_dir = "outputs/vis"
    wmlu.create_empty_dir(save_dir,remove_if_exists=False)
    files = wmlu.recurse_get_filepath_in_dir(file_dir,suffix=".jpg")
    model = Model()
    for file in files:
        print(file)
        img = cv2.imread(file)
        img = model(img)
        save_path = osp.join(save_dir,osp.basename(file))
        cv2.imwrite(save_path,img)
