import os
import cv2
import matplotlib.pyplot as plt
import glob
import argparse
from mmdet.apis import init_detector, inference_detector
from mmyolo.utils import register_all_modules

# Register modules của MMYolo
register_all_modules()

def visualize_detection(config_file, checkpoint_file, img_dir, output_dir, num_imgs=6):
    import torch
    model = init_detector(config_file, checkpoint_file, device="cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Lấy cả .jpg và .png
    img_paths = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
    img_paths = img_paths[:num_imgs]
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(len(axes)):
        if i >= len(img_paths):
            axes[i].axis("off")
            continue
        img_path = img_paths[i]
        result = inference_detector(model, img_path)
        
        from mmdet.registry import VISUALIZERS
        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.dataset_meta = model.dataset_meta
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        visualizer.add_datasample(
            "result",
            img,
            data_sample=result,
            draw_gt=False,
            show=False,
            pred_score_thr=0.3
        )
        drawn_img = visualizer.get_image()
        
        axes[i].imshow(drawn_img)
        axes[i].axis("off")
        axes[i].set_title(os.path.basename(img_path))
        
    plt.tight_layout()
    out_path = os.path.join(output_dir, "Detection_Grid.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    visualize_detection(args.config, args.checkpoint, args.img_dir, args.output_dir)
