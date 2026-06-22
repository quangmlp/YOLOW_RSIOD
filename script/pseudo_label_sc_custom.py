import argparse
import json
import os
import warnings

from mmdet.apis import init_detector
from pycocotools.coco import COCO
from inference import inference_detector
from tqdm import tqdm

warnings.filterwarnings('ignore')


def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return iou


def calculate_max_iou(bbox, bboxs):
    max_iou = 0
    for b in bboxs:
        x1, y1, w, h = b
        x2, y2 = x1 + w, y1 + h
        gt = [x1, y1, x2, y2]
        iou = calculate_iou(bbox, gt)
        if iou > max_iou:
            max_iou = iou
    return max_iou


class Output:
    def __init__(self, xyxy: list, scores: list, cls: list, path):
        self.xyxy = xyxy
        self.scores = scores
        self.cls = cls
        self.path = path


class MmdetModel:
    def __init__(self, cfg_path, pt_path, class_json, skip_scores=0.5) -> None:
        self.cfg_path = cfg_path
        self.pt_path = pt_path
        self.skip_scores = skip_scores
        
        # Load classes from list of lists [[class_name], [class_name]] or list of strings
        class_data = json.load(open(class_json))
        if len(class_data) > 0 and isinstance(class_data[0], list):
            self.class_names = [c[0] for c in class_data]
        else:
            self.class_names = class_data
            
        self.model = init_detector(self.cfg_path, self.pt_path)
        self.class_texts = json.load(open(class_json, 'r'))

    def predict(self, img_path):
        result = inference_detector(self.model, img_path, self.class_texts)
        labels = result.pred_instances.labels
        bboxes = result.pred_instances.bboxes
        scores = result.pred_instances.scores
        ins = scores > self.skip_scores
        bboxes = bboxes[ins, :]
        labels = labels[ins]
        scores = scores[ins]
        return Output(bboxes, scores.tolist(), [self.class_names[cls_idx] for cls_idx in labels], img_path)


def main(mmdet_cfg, mmdet_pt, class_json, ann_file, ann_save_file, img_path, skip_scores, iou_thr):
    model = MmdetModel(mmdet_cfg, mmdet_pt, skip_scores=skip_scores, class_json=class_json)
    coco = COCO(ann_file)

    if args.dataset == 'DIOR':
        all_classes = [
            "airplane", "airport", "bridge", "Expressway-Service-area", "Expressway-toll-station", 
            "harbor", "overpass", "ship", "trainstation", "vehicle",
            "baseballfield", "basketballcourt", "chimney", "dam", "golffield", 
            "groundtrackfield", "stadium", "storagetank", "tenniscourt", "windmill"
        ]
    elif args.dataset == 'DOTA':
        all_classes = [
            "small-vehicle", "large-vehicle", "airplane", "baseball-diamond", "ground-track-field",
            "helicopter", "ship", "bridge", "soccer-ball-field", "tennis-court",
            "storage-tank", "harbor", "roundabout", "basketball-court", "swimming-pool"
        ]
    else:
        raise ValueError("Unknown dataset")
        
    categories_map = {name: i for i, name in enumerate(all_classes)}

    ann_save = json.load(open(ann_file))
    ann_save['categories'] = [{"id": i, "name": name} for i, name in enumerate(all_classes)]

    for ann in ann_save['annotations']:
        ann['score'] = 1.0

    max_id = 0
    if len(ann_save["annotations"]) > 0:
        for ann in ann_save["annotations"]:
            max_id = max(max_id, ann.get("id", 0))

    image_ids = coco.getImgIds()
    print(f"数据集中共有 {len(image_ids)} 张图片")
    for idx, image_id in tqdm(enumerate(image_ids)):
        image_info = coco.loadImgs([image_id])[0]
        image_path = os.path.join(img_path, image_info['file_name'])
        
        ann_ids = coco.getAnnIds(imgIds=[image_info['id']])
        annotations = coco.loadAnns(ann_ids)
        bboxs_gt = [ann['bbox'] for ann in annotations]
        result = model.predict(image_path)

        boxes, types, scores = result.xyxy, result.cls, result.scores
        for box, cls, score in zip(boxes, types, scores):
            box = box.tolist()

            if cls not in categories_map:
                continue

            max_iou = calculate_max_iou(box, bboxs_gt)
            cls = categories_map[cls]
            if max_iou < iou_thr:
                max_id += 1
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                ann = {'image_id': image_id, 'bbox': [x1, y1, w, h], 'category_id': cls, 'id': max_id,
                       'iscrowd': 0,
                       'area': w * h,
                       'segmentation': [[
                           x1, y1,
                           x2, y1,
                           x2, y2,
                           x1, y2
                       ]],
                       'score': score
                       }
                ann_save["annotations"].append(ann)

    with open(ann_save_file, 'w') as f:
        json.dump(ann_save, f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate self-correction pseudo labels for Custom datasets"
    )

    parser.add_argument('--dataset', default='DIOR', type=str,
                        help='Dataset (DIOR or DOTA)')
    parser.add_argument('--stage', type=int, required=True,
                        help='Incremental stage index (e.g., 1 for Task 1)')
    parser.add_argument('--score_thr', type=float, default=0.1)
    parser.add_argument('--iou_thr', type=float, default=0.5)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    stage = args.stage
    prev_stage = stage - 1

    if args.dataset == 'DIOR':
        ann_file = f'/content/data/DIOR/annotation/annotation/train_task_{stage + 1}.json'
        img_path = f'/content/data/DIOR/images/'
        work_dir = f'work_dirs/yolo_iod_dior_10_10_task{prev_stage}'
        class_json = f'/content/data/DIOR/annotation/annotation/dior_class_texts_stage{prev_stage}.json'
    elif args.dataset == 'DOTA':
        ann_file = f'/content/data/DOTA/ann/train_task_{stage + 1}.json'
        img_path = f'/content/data/DOTA/train/images/train/'
        work_dir = f'work_dirs/yolo_iod_dota_5_5_5_task{prev_stage}'
        class_json = f'/content/data/DOTA/ann/dota_class_texts_stage{prev_stage}.json'

    mmdet_cfg = f'{work_dir}/yolo_iod_{args.dataset.lower()}_task{prev_stage}.py'
    if args.dataset == 'DIOR':
        mmdet_cfg = f'configs/dior_10_10/yolo_iod_dior_10_10_task{prev_stage}.py'
    elif args.dataset == 'DOTA':
        mmdet_cfg = f'configs/dota_5_5_5/yolo_iod_dota_5_5_5_task{prev_stage}.py'
        
    mmdet_pt = f'{work_dir}/epoch_20.pth'
    ann_save_file = ann_file.replace('.json', '_ps.json')

    main(
        mmdet_cfg,
        mmdet_pt,
        class_json,
        ann_file,
        ann_save_file,
        img_path,
        args.score_thr,
        args.iou_thr
    )
