import time
import sys
import re

LOG_FILE = "e:/project/三角洲行动物价捕获分析/TimeXer/training19.log"
TOTAL_MODELS = 19

def monitor():
    print(f"开始监视训练日志: {LOG_FILE}")
    print(f"总计需要训练 {TOTAL_MODELS} 个模型。\n")
    
    models_completed = 0
    current_model = ""
    current_epoch = 0
    
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            # Move to the end of file? No, read from beginning to catch up
            f.seek(0)
            
            while True:
                line = f.readline()
                if not line:
                    time.sleep(2)
                    continue
                
                # Check for start of a new model
                if ">>>>>>>start training :" in line:
                    match = re.search(r'long_term_forecast_(.*?)_TimeXer', line)
                    if match:
                        current_model = match.group(1)
                        print(f"\n[{models_completed + 1}/{TOTAL_MODELS}] 开始训练模型: {current_model}")
                
                # Check for epoch progress
                if line.startswith("Epoch: ") and "cost time" not in line:
                    match = re.search(r'Epoch: (\d+).*?Train Loss: ([\d\.]+|nan).*?Vali Loss: ([\d\.]+|nan).*?Test Loss: ([\d\.]+|nan)', line)
                    if match:
                        current_epoch = int(match.group(1))
                        train_loss = match.group(2)
                        vali_loss = match.group(3)
                        test_loss = match.group(4)
                        sys.stdout.write(f"\r  -> Epoch {current_epoch:03d} | Train: {train_loss} | Vali: {vali_loss} | Test: {test_loss}")
                        sys.stdout.flush()
                
                # Check for Early Stopping
                if "Early stopping" in line:
                    print(f"\n  -> 模型 {current_model} 触发早停 (Early Stopping)。")
                
                # Check for end of a model (usually indicated by testing phase or just counting up)
                if ">>>>>>>testing :" in line:
                    print(f"\n[{models_completed + 1}/{TOTAL_MODELS}] 模型 {current_model} 测试完成！")
                    models_completed += 1
                    
                    if models_completed >= TOTAL_MODELS:
                        print("\n🎉 所有 19 个模型均已训练并测试完成！")
                        break
                        
    except KeyboardInterrupt:
        print("\n监视已手动中止。")
    except Exception as e:
        print(f"\n监视发生错误: {e}")

if __name__ == "__main__":
    monitor()
