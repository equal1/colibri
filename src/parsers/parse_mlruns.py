import os
import yaml
import json

def extract_value(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        return [float(line.split()[1]) for line in lines]

def extract_run_data_from_meta(meta_path):
    with open(meta_path, 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)

def extract_metrics_from_folder(metrics_folder):
    metrics = {}
    for metric_file in ["train_accuracy_epoch", "train_loss_epoch", "val_accuracy_epoch", "val_loss_epoch"]:
        if os.path.exists(os.path.join(metrics_folder, metric_file)):
            metrics[metric_file] = extract_value(os.path.join(metrics_folder, metric_file))
    return metrics

def parse_mlruns(dir_path):
    runs_data = []
    for root, dirs, files in os.walk(dir_path):
        if root.endswith("/tags"):
            run_data = extract_run_data_from_meta(os.path.join(root, "..", "meta.yaml"))
            run_data.update(extract_metrics_from_folder(os.path.join(root, "..", "metrics")))
            runs_data.append(run_data)

    # Sort the data by the number at the end of run_name
    return sorted(runs_data, key=lambda x: int(x['start_time']))

def save_to_json(data, save_path):
    with open(save_path, "w") as file:
        json.dump(data, file, indent=4)

def parse_mlruns(mlruns_path="mlruns", save_path="mlruns_report.json"):
    data = parse_mlruns(mlruns_path)
    save_to_json(data, save_path)

