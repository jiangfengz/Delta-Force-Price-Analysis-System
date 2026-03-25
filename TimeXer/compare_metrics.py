import os
import glob
import pandas as pd

def parse_metrics(csv_path):
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        if not df.empty and 'MSE' in df.columns:
            return float(df['MSE'].iloc[0]), float(df['MAE'].iloc[0])
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    return None

def extract_bullet_name(folder_name):
    # e.g., V17_7d_4570Govt_Collection_... -> 4570Govt
    parts = folder_name.split('_')
    if len(parts) > 2:
        return parts[2]
    return folder_name

def main():
    backup_dir = r"e:\project\三角洲行动物价捕获分析\TimeXer\backup\best in 7 days"
    results_dir = r"e:\project\三角洲行动物价捕获分析\TimeXer\results"
    
    # 1. Gather backup metrics
    backup_metrics = {}
    for folder in os.listdir(backup_dir):
        full_path = os.path.join(backup_dir, folder)
        if os.path.isdir(full_path):
            bullet = extract_bullet_name(folder)
            csv_path = os.path.join(full_path, "results", "metrics.csv")
            # If the structure is directly under the folder (some versions might vary)
            if not os.path.exists(csv_path):
                # check subfolders
                sub_dirs = glob.glob(os.path.join(full_path, "*", "metrics.csv"))
                if sub_dirs:
                    csv_path = sub_dirs[0]
            
            metrics = parse_metrics(csv_path)
            if metrics:
                backup_metrics[bullet] = metrics
                
    # 2. Gather result metrics
    result_metrics = {}
    for folder in os.listdir(results_dir):
        full_path = os.path.join(results_dir, folder)
        if os.path.isdir(full_path) and folder.startswith("V19_"):
            bullet = extract_bullet_name(folder)
            csv_path = os.path.join(full_path, "metrics.csv")
            metrics = parse_metrics(csv_path)
            if metrics:
                result_metrics[bullet] = metrics
                
    # 3. Compare
    print(f"{'Bullet Type':<15} | {'Old MSE':<10} | {'New MSE':<10} | {'Old MAE':<10} | {'New MAE':<10} | {'Result':<10}")
    print("-" * 75)
    
    better_count = 0
    worse_count = 0
    
    bullets = set(backup_metrics.keys()).union(set(result_metrics.keys()))
    for bullet in sorted(bullets):
        old = backup_metrics.get(bullet)
        new = result_metrics.get(bullet)
        
        old_mse = f"{old[0]:.5f}" if old else "N/A"
        old_mae = f"{old[1]:.5f}" if old else "N/A"
        new_mse = f"{new[0]:.5f}" if new else "N/A"
        new_mae = f"{new[1]:.5f}" if new else "N/A"
        
        result = "---"
        if old and new:
            if new[0] < old[0]:
                result = "✅ BETTER"
                better_count += 1
            else:
                result = "❌ WORSE"
                worse_count += 1
        
        print(f"{bullet:<15} | {old_mse:<10} | {new_mse:<10} | {old_mae:<10} | {new_mae:<10} | {result}")
        
    print("-" * 75)
    print(f"Summary: {better_count} models improved, {worse_count} models worsened.")

if __name__ == "__main__":
    main()
