# Colibri: A Framework for Automated Quantum Dot Calibration

## Overview
Colibri, named after the hummingbird genus and serving as a playful pun on "calibration," aims to provide a comprehensive solution for the calibration of quantum dots. Recognizing the challenges and limitations of manual tuning and script-based approaches, our planned framework leverages machine learning algorithms to automate the tasks of creation, initialization, and control of quantum dots-critical steps for advancing scalability.

## Project Structure
The project is structured as follows:

```
├── data                  <- Data for use in this project.Only the sorted data should be stored here.
|   |
│   ├── external          <- Data from third party sources.
│   ├── interim           <- Intermediate data that has been transformed.
│   ├── processed         <- The final, canonical data sets for modeling.
│   └── raw               <- The original, immutable data dump.
│   
├── docs                  <- Directory for documentation related to the project
│   
├── experiments           <- Scripts for running experiments
│  └── databases          <- All the .db files created by running the experiment scripts
│   
├── models                <- Trained models, model checkpoints, or model summaries
│   
├── notebooks             <- Jupyter notebooks. Naming convention is a number (for ordering),
│                           the creator's initials, and a short `-` delimited description, e.g.
│                           `1.0-jqp-initial-data-exploration`.
│
├── references            <- Papers, articles, and all other explanatory materials.
│   
├── reports               <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures           <- Generated graphics and figures to be used in reporting
|
├── setup_scripts         <- Scripts for setting up the environment
│   
├── src                   <- Source code for use in this project.
│   ├── __init__.py       <- Makes src a Python module
│   │   
│   ├── colibri           <- Root namespace for the framework
│   │
│   └── machine_learning  <- Directory for machine learning related code
│     ├── data            <- Scripts to download or generate data
│     │
│     ├── features        <- Scripts to turn raw data into features for modeling
│     │
│     ├── models          <- Scripts to train models and then use trained models to make
│     │                      predictions
│     │
│     └── visualization   <- Scripts to create exploratory and results oriented visualizations
│
├── tests                 <- Directory for unit tests
|
├── .gitignore            <- Files and directories to be ignored by git
│
├── README.md             <- The top-level README for developers using this project.
│
├── requirements.txt      <- The requirements file for reproducing the analysis environment, e.g.
│                            generated with `pip freeze > requirements.txt`
│
│
└── setup.py              <- Make this project pip installable with `pip install -e`(TODO)

```
## Resources

### Recommended Reading for Understanding the Problem
Below is a list of papers and resources that offer valuable insights into the challenges of quantum dot calibration:

1. **Colloquium: Advances in automation of quantum dot devices control**: This paper provides an excellent starting point by detailing the current state of calibration techniques and existing solutions.  
   **Link**: [Colloquium: Advances in automation of quantum dot devices control](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10088060/#S8)

---

Note: Anyone who contributes to this repo can expand this README.md file with any information that they deem useful for the users or other developers.
