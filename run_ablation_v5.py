import os
os.environ['MPLBACKEND'] = 'agg'
import matplotlib
matplotlib.use('Agg')
import glob
import re
import matplotlib.pyplot as plt

# --- CAU HINH QUAN TRONG ---
# FAST_MODE = True se ep xung chu trinh huan luyen xuong 10 epoch dong bo, tiet kiem 50% thoi gian
FAST_MODE = True

BASE_OPTS = ""
if FAST_MODE:
    # Ep max_epochs = 10, LR scheduler = 10, Tat Mosaic = 5, Chi Evaluate = 10
    BASE_OPTS = "--cfg-options train_cfg.max_epochs=10 default_hooks.param_scheduler.max_epochs=10 custom_hooks.1.switch_epoch=5 train_cfg.val_interval=10 train_cfg.dynamic_intervals=[] "

CMD_DIOR_T1 = 'MPLBACKEND=agg PYTHONPATH="$(pwd):$(pwd)/third_party/mmyolo" bash tools/dist_train.sh configs/dior_10_10/yolo_iod_dior_10_10_task1.py 1 --amp --work-dir work_dirs_ablation/{} '
CMD_DOTA_T1 = 'MPLBACKEND=agg PYTHONPATH="$(pwd):$(pwd)/third_party/mmyolo" bash tools/dist_train.sh configs/dota_5_5_5/yolo_iod_dota_5_5_5_task1.py 1 --amp --work-dir work_dirs_ablation/{} '
CMD_DOTA_T2 = 'MPLBACKEND=agg PYTHONPATH="$(pwd):$(pwd)/third_party/mmyolo" bash tools/dist_train.sh configs/dota_5_5_5/yolo_iod_dota_5_5_5_task2.py 1 --amp --work-dir work_dirs_ablation/{} '

# The epoch to load from Task 1 (depends on FAST_MODE)
LOAD_EPOCH = 10 if FAST_MODE else 20

experiments = [
    # === TABLE 4: Components Ablation (DOTA 5+5+5 Task 2 SEQUENTIAL) ===
    # Task 1
    {"name": "dota_t1_cpr_only", "cmd": CMD_DOTA_T1, "opts": "train_cfg.gps_ratio=0.0 train_cfg.up_ratio=0.0 model.kd_cfg.kd_new=False model.kd_cfg.loss_cls_kd_temperature_old=0 model.kd_cfg.loss_reg_kd_temperature_old=0"},
    {"name": "dota_t1_cpr_iks", "cmd": CMD_DOTA_T1, "opts": "model.kd_cfg.kd_new=False model.kd_cfg.loss_cls_kd_temperature_old=0 model.kd_cfg.loss_reg_kd_temperature_old=0"},
    {"name": "dota_t1_cpr_cakd", "cmd": CMD_DOTA_T1, "opts": "train_cfg.gps_ratio=0.0 train_cfg.up_ratio=0.0"},
    # Task 2
    {"name": "dota_t2_cpr_only", "cmd": CMD_DOTA_T2, "opts": f"train_cfg.gps_ratio=0.0 train_cfg.up_ratio=0.0 model.kd_cfg.kd_new=False model.kd_cfg.loss_cls_kd_temperature_old=0 model.kd_cfg.loss_reg_kd_temperature_old=0 load_from=work_dirs_ablation/dota_t1_cpr_only/epoch_{LOAD_EPOCH}.pth"},
    {"name": "dota_t2_cpr_iks", "cmd": CMD_DOTA_T2, "opts": f"model.kd_cfg.kd_new=False model.kd_cfg.loss_cls_kd_temperature_old=0 model.kd_cfg.loss_reg_kd_temperature_old=0 load_from=work_dirs_ablation/dota_t1_cpr_iks/epoch_{LOAD_EPOCH}.pth"},
    {"name": "dota_t2_cpr_cakd", "cmd": CMD_DOTA_T2, "opts": f"train_cfg.gps_ratio=0.0 train_cfg.up_ratio=0.0 load_from=work_dirs_ablation/dota_t1_cpr_cakd/epoch_{LOAD_EPOCH}.pth"},
    # dota_t2_full is already in FIGURE 3
    
    # === FIGURE 3: CAKD Ablation (DOTA 5+5+5 Task 1 & Task 2 SEQUENTIAL) ===
    # Task 1
    {"name": "dota_t1_cakd_old", "cmd": CMD_DOTA_T1, "opts": "model.kd_cfg.kd_new=False"},
    {"name": "dota_t1_cakd_cur", "cmd": CMD_DOTA_T1, "opts": "model.kd_cfg.loss_cls_kd_temperature_old=0 model.kd_cfg.loss_reg_kd_temperature_old=0"},
    {"name": "dota_t1_full", "cmd": CMD_DOTA_T1, "opts": ""},
    # Task 2 (Tuần tự load trọng số từ bản Task 1 tương ứng)
    {"name": "dota_t2_cakd_old", "cmd": CMD_DOTA_T2, "opts": f"model.kd_cfg.kd_new=False load_from=work_dirs_ablation/dota_t1_cakd_old/epoch_{LOAD_EPOCH}.pth"},
    {"name": "dota_t2_cakd_cur", "cmd": CMD_DOTA_T2, "opts": f"model.kd_cfg.loss_cls_kd_temperature_old=0 model.kd_cfg.loss_reg_kd_temperature_old=0 load_from=work_dirs_ablation/dota_t1_cakd_cur/epoch_{LOAD_EPOCH}.pth"},
    {"name": "dota_t2_full", "cmd": CMD_DOTA_T2, "opts": f"load_from=work_dirs_ablation/dota_t1_full/epoch_{LOAD_EPOCH}.pth"},
    
    # === FIGURE 4: IKS Ratio K (DOTA 5+5+5 Task 1 & Task 2 SEQUENTIAL) ===
    # Task 1
    {"name": "dota_t1_iks_k100", "cmd": CMD_DOTA_T1, "opts": "train_cfg.gps_ratio=0.0 train_cfg.up_ratio=0.0"},
    {"name": "dota_t1_iks_k12", "cmd": CMD_DOTA_T1, "opts": "train_cfg.gps_ratio=0.88 train_cfg.up_ratio=0.88"},
    {"name": "dota_t1_iks_k8", "cmd": CMD_DOTA_T1, "opts": "train_cfg.gps_ratio=0.92 train_cfg.up_ratio=0.92"},
    {"name": "dota_t1_iks_k5", "cmd": CMD_DOTA_T1, "opts": "train_cfg.gps_ratio=0.95 train_cfg.up_ratio=0.95"},
    # Task 2 (Tuần tự load trọng số từ bản Task 1 tương ứng)
    {"name": "dota_t2_iks_k100", "cmd": CMD_DOTA_T2, "opts": f"train_cfg.gps_ratio=0.0 train_cfg.up_ratio=0.0 load_from=work_dirs_ablation/dota_t1_iks_k100/epoch_{LOAD_EPOCH}.pth"},
    {"name": "dota_t2_iks_k12", "cmd": CMD_DOTA_T2, "opts": f"train_cfg.gps_ratio=0.88 train_cfg.up_ratio=0.88 load_from=work_dirs_ablation/dota_t1_iks_k12/epoch_{LOAD_EPOCH}.pth"},
    {"name": "dota_t2_iks_k8", "cmd": CMD_DOTA_T2, "opts": f"train_cfg.gps_ratio=0.92 train_cfg.up_ratio=0.92 load_from=work_dirs_ablation/dota_t1_iks_k8/epoch_{LOAD_EPOCH}.pth"},
    {"name": "dota_t2_iks_k5", "cmd": CMD_DOTA_T2, "opts": f"train_cfg.gps_ratio=0.95 train_cfg.up_ratio=0.95 load_from=work_dirs_ablation/dota_t1_iks_k5/epoch_{LOAD_EPOCH}.pth"},
]

def parse_log(work_dir):
    """Trích xuất kết quả mAP từ file log mới nhất"""
    log_files = glob.glob(os.path.join(work_dir, '*', '*.log'))
    if not log_files: return None
    latest_log = max(log_files, key=os.path.getctime)
    with open(latest_log, 'r') as f:
        content = f.read()
    
    matches = re.findall(r'coco/bbox_mAP:\s+(\d+\.\d+).*?coco/bbox_mAP_50:\s+(\d+\.\d+)', content)
    if matches:
        return float(matches[-1][0]) * 100, float(matches[-1][1]) * 100
    return None

def run_all():
    print(f"🚀 BẮT ĐẦU CHẠY ABLATION STUDY V5 (FAST_MODE = {FAST_MODE}, SEQUENTIAL LEARNING)...")
    print(f"Tổng số kịch bản tuần tự: {len(experiments)}")
    for i, exp in enumerate(experiments):
        name = exp["name"]
        work_dir = f"work_dirs_ablation/{name}"
        if os.path.exists(os.path.join(work_dir, 'last_checkpoint')):
            print(f"[{i+1}/{len(experiments)}] ⏩ Đã tìm thấy kết quả của {name}, bỏ qua train.")
        else:
            print(f"[{i+1}/{len(experiments)}] 🔄 Đang train kịch bản: {name}")
            os.makedirs(work_dir, exist_ok=True)
            cmd = exp["cmd"].format(name) + BASE_OPTS + exp["opts"]
            cmd = f"{cmd} 2>&1 | tee {work_dir}/console.log"
            print("Chạy lệnh:", cmd)
            os.system(cmd)
            
def parse_dota_t2_log_split(work_dir):
    log_files = glob.glob(os.path.join(work_dir, '*', '*.log'))
    if not log_files: return None
    latest_log = max(log_files, key=os.path.getctime)
    with open(latest_log, 'r') as f:
        content = f.read()
        
    matches = re.findall(r'coco/([a-zA-Z0-9\-_]+)_precision:\s+(\d+\.\d+)', content)
    if not matches: return None
    
    matches = matches[-15:]
    if len(matches) != 15: return None
    
    aps = [float(m[1]) * 100 for m in matches]
    base_ap = sum(aps[:5]) / 5.0
    inc1_ap = sum(aps[5:10]) / 5.0
    inc2_ap = sum(aps[10:]) / 5.0
    all_ap = sum(aps) / 15.0
    return base_ap, inc1_ap, inc2_ap, all_ap

def plot_results():
    results = {}
    for exp in experiments:
        name = exp["name"]
        res = parse_log(f"work_dirs_ablation/{name}")
        if res: results[name] = res
        else: print(f"⚠️ Chưa có kết quả cho {name}")

    dota_t2_splits = {}
    for name in ["dota_t2_cpr_only", "dota_t2_cpr_iks", "dota_t2_cpr_cakd", "dota_t2_full"]:
        split_res = parse_dota_t2_log_split(f"work_dirs_ablation/{name}")
        if split_res: dota_t2_splits[name] = split_res

    import numpy as np

    # 1. Vẽ biểu đồ CAKD (Figure 3 - Line Chart)
    try:
        base_res = parse_log("work_dirs/yolo_iod_dota_5_5_5_task0")
        base_mAP = base_res[0] if base_res else 0.0
        
        labels = ['Task 0 (Base)', 'Task 1', 'Task 2']
        cakd_old = [base_mAP, results.get('dota_t1_cakd_old', [0])[0], results.get('dota_t2_cakd_old', [0])[0]]
        cakd_cur = [base_mAP, results.get('dota_t1_cakd_cur', [0])[0], results.get('dota_t2_cakd_cur', [0])[0]]
        cakd_full = [base_mAP, results.get('dota_t1_full', [0])[0], results.get('dota_t2_full', [0])[0]]
        
        plt.figure(figsize=(6, 4))
        plt.plot(labels, cakd_old, marker='o', label='CAKD (Old)', color='#FFA07A', linewidth=2)
        plt.plot(labels, cakd_cur, marker='^', label='CAKD (Cur)', color='#20B2AA', linewidth=2)
        plt.plot(labels, cakd_full, marker='s', label='CAKD (Full)', color='#00CED1', linewidth=2)
        plt.title('Ablation of CAKD Module (DOTA-IOD)')
        plt.ylabel('mAP (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('work_dirs_ablation/fig3_cakd_ablation.png', bbox_inches='tight', dpi=300)
        print("✅ Đã lưu work_dirs_ablation/fig3_cakd_ablation.png")
    except Exception as e:
        print("Lỗi vẽ Figure 3:", e)

    # 2. Vẽ biểu đồ IKS Ratio K (Figure 4 - Grouped Bar Chart)
    try:
        x_labels = ['5%', '8%', '12%', '20%', '100%']
        t1_maps = [
            results.get('dota_t1_iks_k5', [0])[0],
            results.get('dota_t1_iks_k8', [0])[0],
            results.get('dota_t1_iks_k12', [0])[0],
            results.get('dota_t1_full', [0])[0], # K=20%
            results.get('dota_t1_iks_k100', [0])[0]
        ]
        t2_maps = [
            results.get('dota_t2_iks_k5', [0])[0],
            results.get('dota_t2_iks_k8', [0])[0],
            results.get('dota_t2_iks_k12', [0])[0],
            results.get('dota_t2_full', [0])[0], # K=20%
            results.get('dota_t2_iks_k100', [0])[0]
        ]
        
        x = np.arange(len(x_labels))
        width = 0.35
        
        plt.figure(figsize=(8, 5))
        bars1 = plt.bar(x - width/2, t1_maps, width, label='Task 1', color='#DB7093')
        bars2 = plt.bar(x + width/2, t2_maps, width, label='Task 2', color='#FFB6C1')
        
        plt.title('Ablation of Kernel Selection Ratio K (DOTA-IOD)')
        plt.xlabel('K Ratio')
        plt.ylabel('mAP (%)')
        plt.xticks(x, x_labels)
        plt.legend()
        
        for bar in bars1 + bars2:
            yval = bar.get_height()
            if yval > 0:
                plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{yval:.1f}", ha='center', va='bottom', fontsize=8)
                
        plt.savefig('work_dirs_ablation/fig4_iks_ablation.png', bbox_inches='tight', dpi=300)
        print("✅ Đã lưu work_dirs_ablation/fig4_iks_ablation.png")
    except Exception as e:
        print("Lỗi vẽ Figure 4:", e)

    # 3. In ra bảng Component (Table 4)
    print("\n" + "="*70)
    print("📊 TABLE 4: ABLATION COMPONENTS (DOTA-IOD TASK 2)")
    print("="*70)
    print(f"{'CPR':<5} | {'IKS':<5} | {'CAKD':<5} | {'1-5 (Base)':<10} | {'6-10 (Inc1)':<11} | {'11-15 (Inc2)':<12} | {'1-15 (All)':<10}")
    print("-" * 70)
    
    def print_row(cpr, iks, cakd, name):
        res = dota_t2_splits.get(name, (0.0, 0.0, 0.0, 0.0))
        print(f"{cpr:<5} | {iks:<5} | {cakd:<5} | {res[0]:<10.1f} | {res[1]:<11.1f} | {res[2]:<12.1f} | {res[3]:<10.1f}")
        
    print_row("✓", "", "", "dota_t2_cpr_only")
    print_row("✓", "✓", "", "dota_t2_cpr_iks")
    print_row("✓", "", "✓", "dota_t2_cpr_cakd")
    print_row("✓", "✓", "✓", "dota_t2_full")
    print("="*70)

if __name__ == '__main__':
    os.makedirs('work_dirs_ablation', exist_ok=True)
    run_all()
    plot_results()
