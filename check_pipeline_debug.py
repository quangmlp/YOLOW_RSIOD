import sys
import os
import inspect
from mmengine.config import Config
from mmengine.registry import init_default_scope

def main():
    init_default_scope('mmyolo')
    
    from mmyolo.registry import TRANSFORMS
    load_ann_cls = TRANSFORMS.get('LoadAnnotations')
    print("=== LoadAnnotations Resolution ===")
    print("Class:", load_ann_cls)
    try:
        print("File Path:", inspect.getfile(load_ann_cls))
    except Exception as e:
        print("Could not get file path:", e)
    print("==================================\n")
    
    config_path = 'configs/dior_10_10/yolo_iod_dior_10_10_task0.py'
    if not os.path.exists(config_path):
        print(f"Error: Cannot find config file at {config_path}")
        return
        
    cfg = Config.fromfile(config_path)
    import yolo_world
    from mmyolo.registry import DATASETS
    
    dataset_cfg = cfg.dior_train_dataset.copy()
    dataset_cfg.pop('_delete_', None)
    if 'dataset' in dataset_cfg and 'filter_cfg' in dataset_cfg['dataset']:
        dataset_cfg['dataset']['filter_cfg']['min_size'] = 1
        
    dataset = DATASETS.build(dataset_cfg)
    print("Dataset built successfully!")
    
    idx = 0
    data_info = dataset.get_data_info(idx)
    
    print("\n=== Initial Raw Info ===")
    print("Instances (first 5):")
    for i, inst in enumerate(data_info.get('instances', [])[:5]):
        print(f"  - Inst {i}: bbox={inst.get('bbox')}, label={inst.get('bbox_label')}, ignore_flag={inst.get('ignore_flag')}")
            
    # Run the pipeline step-by-step
    data = data_info.copy()
    data['dataset'] = dataset
    
    print("\n=== Running Pipeline Step-by-Step ===")
    for i, transform in enumerate(dataset.pipeline.transforms):
        transform_name = transform.__class__.__name__
        
        # If this is LoadAnnotations, let's trace it manually!
        if transform_name == 'LoadAnnotations':
            print(f"\n[Debug] Manually tracing LoadAnnotations on data['instances'] (len={len(data.get('instances', []))})")
            gt_bboxes = []
            gt_ignore_flags = []
            for instance in data.get('instances', []):
                gt_bboxes.append(instance['bbox'])
                gt_ignore_flags.append(instance['ignore_flag'])
            print(f"[Debug] Manually built gt_bboxes (len={len(gt_bboxes)}): {gt_bboxes}")
            print(f"[Debug] Manually built gt_ignore_flags (len={len(gt_ignore_flags)}): {gt_ignore_flags}")
            
        try:
            data = transform(data)
            print(f"\nStep {i}: {transform_name}")
            if data is None:
                print("  returned None!")
                break
            print("  Keys:", list(data.keys()))
            if 'gt_bboxes' in data:
                print(f"  gt_bboxes (len={len(data['gt_bboxes'])}):", data['gt_bboxes'])
            if 'gt_bboxes_labels' in data:
                print(f"  gt_bboxes_labels (len={len(data['gt_bboxes_labels'])}):", data['gt_bboxes_labels'])
            if 'instances' in data:
                print(f"  instances (len={len(data['instances'])}):")
                for j, inst in enumerate(data['instances'][:5]):
                    print(f"    - Inst {j}: bbox={inst.get('bbox')}, label={inst.get('bbox_label')}, ignore_flag={inst.get('ignore_flag')}")
        except Exception as e:
            print(f"\nError at Step {i} ({transform_name}): {e}")
            import traceback
            traceback.print_exc()
            break

if __name__ == '__main__':
    main()
