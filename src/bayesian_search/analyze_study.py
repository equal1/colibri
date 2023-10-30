import os

import pickle
import plotly.io as pio


from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_slice
from optuna.visualization import plot_param_importances

from search.parsers import load_study_from_pickle


def display_study_info(study):
    # Print study name and direction
    print(f"Study name: {study.study_name}")
    print(f"Direction: {study.direction}")

    # Print best trial
    best_trial = study.best_trial
    print(f"Best trial: #{best_trial.number}")
    print(f"  Value: {best_trial.value}")
    print("  Params:")

    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Print number of trials
    print(f"Number of trials: {len(study.trials)}")


def visualize_study(save_dir, study):
    # Ensure save directory exists
    study_dir = os.path.join(save_dir, study.study_name)
    if not os.path.exists(study_dir):
        os.makedirs(study_dir)

    # Plot optimization history
    fig = plot_optimization_history(study)
    pio.write_html(fig, f"{study_dir}/optimization_history.html")

    # Plot parallel coordinates
    fig = plot_parallel_coordinate(study)
    pio.write_html(fig, f"{study_dir}/parallel_coordinate.html")

    # Plot slice plot
    fig = plot_slice(study)
    pio.write_html(fig, f"{study_dir}/slice_plot.html")

    # Plot parameter importance
    fig = plot_param_importances(study)
    pio.write_html(fig, f"{study_dir}/parameter_importance.html")

def main():
    filename = 'logs/study_state_estimator.pkl'
    study = load_study_from_pickle(filename)

    display_study_info(study)
    visualize_study('figures/study', study)

if __name__ == "__main__":
    main()
