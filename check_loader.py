import sys
import os
import json
import collections
from mmengine.config import Config
from mmengine.registry import init_default_scope

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check dataset loader and category mappings")
    parser.add_argument('--config', type=str, default='configs/dior_10_10/yolo_iod_dior_10_10_task0.py',
                        help='Path to the configuration file to test')
    args = parser.parse_args()

    # Khởi tạo scope mmyolo thay vì mmdet
    init_default_scope('mmyolo')
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: Cannot find config file at {config_path}")
        return

    print(f"Loading config from {config_path}...")
    cfg = Config.fromfile(config_path)

    try:
        import yolo_world
        print("Successfully imported yolo_world modules.")
    except Exception as e:
        sys.path.insert(0, os.getcwd())
        import yolo_world
        print("Successfully imported yolo_world after adding CWD to path.")

    # Import DATASETS từ mmyolo
    from mmyolo.registry import DATASETS

    print("Building train dataset...")
    try:
        dataset_cfg = cfg.dior_train_dataset.copy()
        dataset_cfg.pop('_delete_', None)  # Loại bỏ khóa kế thừa _delete_ để tránh TypeError
        # Ép min_size = 1 để tránh lọc mất các box nhỏ đặc trưng của ảnh vệ tinh
        if 'dataset' in dataset_cfg and 'filter_cfg' in dataset_cfg['dataset']:
            dataset_cfg['dataset']['filter_cfg']['min_size'] = 1
        dataset = DATASETS.build(dataset_cfg)
        print("Dataset built successfully!")
    except Exception as e:
        print(f"Failed to build dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # Lấy thông tin lớp và ID bằng phương thức chính thống của MMDetection 3.x
    dataset_classes = dataset.dataset.metainfo.get('classes', [])
    cat2label = getattr(dataset.dataset, 'cat2label', {})

    print("\n--- Categories mapped in dataset ---")
    for i, cat_id in enumerate(dataset.dataset.cat_ids):
        print(f"  Category ID {cat_id:2d}: '{dataset_classes[i]}'")

    print("\n--- Dataset Metainfo Classes (classes list) ---")
    for idx, name in enumerate(dataset_classes):
        print(f"  Index {idx:2d}: '{name}'")

    # Check cat_ids list in CocoDataset
    print("\n--- Category ID Mapping in CocoDataset ---")
    print(f"dataset.cat_ids: {dataset.dataset.cat_ids}")
    
    # Check internal cat2label mapping
    print(f"dataset.cat2label mapping:")
    for cat_id, label_idx in sorted(cat2label.items()):
        print(f"  Category ID {cat_id:2d} -> Internal Label {label_idx}")

    # Count known vs unknown categories in the entire dataset
    print("\n--- Scanning dataset annotations for labels ---")
    known_count = 0
    unknown_count = 0
    total_boxes = 0
    
    label_distribution = collections.defaultdict(int)

    for idx in range(len(dataset)):
        raw_info = dataset.get_data_info(idx)
        instances = raw_info.get('instances', [])
        for inst in instances:
            label_idx = inst.get('bbox_label')
            total_boxes += 1
            
            if label_idx is not None:
                label_distribution[label_idx] += 1
                if label_idx < cfg.num_known_cls:
                    known_count += 1
                else:
                    unknown_count += 1
            else:
                label_distribution[-1] += 1

    print(f"Total annotations found: {total_boxes}")
    print(f"Mapped to known labels (< {cfg.num_known_cls}): {known_count} boxes")
    print(f"Mapped to unknown labels (>= {cfg.num_known_cls}): {unknown_count} boxes")
        
    print("\n--- Annotation distribution by internal Label Index ---")
    for label_idx, count in sorted(label_distribution.items()):
        name = dataset_classes[label_idx] if label_idx >= 0 and label_idx < len(dataset_classes) else 'N/A'
        cat_id = dataset.dataset.cat_ids[label_idx] if label_idx >= 0 and label_idx < len(dataset.dataset.cat_ids) else -1
        print(f"  Label Index {label_idx:2d} (Category ID {cat_id:2d}, {name}): {count} boxes")

    print("\n--- Testing Pipeline Processing on first 5 samples ---")
    for idx in range(min(5, len(dataset))):
        raw_info = dataset.get_data_info(idx)
        raw_instances = raw_info.get('instances', [])
        proc_data = dataset[idx]
        data_sample = proc_data['data_samples']
        labels = data_sample.gt_instances.labels
        texts = data_sample.texts
        
        print(f"\nSample {idx:d}:")
        print(f"  Raw annotations:")
        for i_idx, inst in enumerate(raw_instances):
            label_idx = inst.get('bbox_label')
            name = dataset_classes[label_idx] if label_idx is not None else 'N/A'
            bbox = inst.get('bbox')
            print(f"    - Box {i_idx}: class='{name}' (label_idx={label_idx}), bbox={bbox}")
            
        print(f"  texts in batch input (len={len(texts)}): {texts}")
        print(f"  gt_instances.labels: {labels.tolist()}")
        for l_idx in labels.tolist():
            class_name = texts[l_idx] if l_idx < len(texts) else 'OUT_OF_BOUNDS!'
            print(f"    Label {l_idx} -> Text: '{class_name}'")

if __name__ == '__main__':
    main()
