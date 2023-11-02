import pickle

def load_study_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def parse_trial_log_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    trials = []
    trial = {}
    for line in lines:
        if line.startswith("Trial #"):
            if trial:
                trials.append(trial)
                trial = {}
            trial['Trial'] = int(line.split("#")[-1])
        elif line.startswith("  Value: "):
            trial['Value'] = float(line.split(": ")[-1])
        elif line.startswith("  Params:"):
            params = {}
        elif line.startswith("    "):
            key, value = line.strip().split(": ")
            if '.' in value:
                params[key] = float(value)
            elif value.isdigit():
                params[key] = int(value)
            else:
                params[key] = value
            trial['Params'] = params

    # Add the last trial
    if trial:
        trials.append(trial)
    
    # Filter out trials with Value less than 0.9
    filtered_trials = [trial for trial in trials if trial['Value'] > 0.9]
    return filtered_trials

def extract_information_from_trials(study_path, log_file='extract_trials_log.txt', order_by='value'):
    study = load_study_from_pickle(study_path) 
    all_trials = study.trials

    # Open the log file in write mode
    with open(log_file, 'w') as log_f:
        # Redirect standard output to the log file

        # Sort the trials by the value
        if order_by == 'value':
            all_trials.sort(key=lambda trial: trial.value, reverse=True)

        for trial in all_trials:
            log_f.write(f"Trial #{trial.number}\n")
            log_f.write(f"  Value: {trial.value}\n")
            log_f.write(f"  Epochs: {len(trial.intermediate_values)}\n")
            log_f.write("  Params:\n")
            for key, value in trial.params.items():
                log_f.write(f"    {key}: {value}\n")
            log_f.write("\n")