import argparse
import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (DatasetCatalog, MetadataCatalog,
                             build_detection_test_loader)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.evaluation.coco_evaluation import COCOEvaluator


def make_config(model_path: str):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg.INPUT.MASK_FORMAT='bitmask'

    cfg.DATASETS.TRAIN = ("asl_train",)
    cfg.DATASETS.TEST = ("asl_val",)
    cfg.TEST.EVAL_PERIOD = 1_000 # eval every 1_000 iterations

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_path

    cfg.OUTPUT_DIR = "./output_segmentation"
    cfg.SOLVER.CHECKPOINT_PERIOD = 1_000

    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
    cfg.INPUT.FORMAT = "RGB"

    cfg.SOLVER.IMS_PER_BATCH = 5  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 15_000    
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 24  

    return cfg

# Trainer for online evaluation
class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator("asl_val",)

def main(args):
    path: str = args.path

    # Register Dataset
    register_coco_instances("asl_train", {}, "./datasets/ASL_mask/annotations/instances_Train.json", "./datasets/ASL_mask/images")
    register_coco_instances("asl_val", {}, "./datasets/ASL_mask/annotations/instances_Test.json", "./datasets/ASL_mask/images")

    cfg = make_config(path)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Evaluation
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("asl_val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "asl_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=False)

    args = parser.parse_args()

    main(args)