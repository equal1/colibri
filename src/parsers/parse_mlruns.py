import os
import yaml
import json

def extract_value(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        return [float(line.split()[1]) for line in lines]

def parse_mlruns(dir_path):
    runs_data = []
    for root, dirs, files in os.walk(dir_path):
        if root.endswith("/tags"):
            run_data = {}
            # Extracting info from meta.yaml
            with open(os.path.join(root, "..", "meta.yaml"), 'r') as file:
                meta_data = yaml.load(file, Loader=yaml.FullLoader)
                run_data.update(meta_data)
            
            # Extracting metrics
            metrics_folder = os.path.join(root, "..", "metrics")
            for metric_file in ["train_accuracy_epoch", "train_loss_epoch", "val_accuracy_epoch", "val_loss_epoch"]:
                if os.path.exists(os.path.join(metrics_folder, metric_file)):
                    run_data[metric_file] = extract_value(os.path.join(metrics_folder, metric_file))
            
            runs_data.append(run_data)

    # Sort the data by the number at the end of run_name
    sorted_runs_data = sorted(runs_data, key=lambda x: int(x['start_time']))

    return sorted_runs_data

data = parse_mlruns("mlruns")
with open("mlruns_report.json", "w") as file:
    json.dump(data, file, indent=4)

print("Parsing completed and saved to output.json!")