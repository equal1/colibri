# Colibri: A Framework for Automated Quantum Dot Calibration

## Overview
Colibri, named after the hummingbird genus and serving as a playful pun on "calibration," aims to provide a comprehensive solution for the calibration of quantum dots. Recognizing the challenges and limitations of manual tuning and script-based approaches, our planned framework leverages machine learning algorithms to automate the tasks of creation, initialization, and control of quantum dots-critical steps for advancing scalability.

Special thanks to Peter and [his project](https://github.com/equal1/eq1x-scripts) for providing a template for both the project structure and the README.md file.

## Project Structure
The project is structured as follows:

```
├── data                  <- Data for use in this project.Only the sorted data should be stored here.
|   |
│   ├── external          <- Links to external data sources.
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
├── resources             <- Any non-code resources that are useful for the project 
│   ├── reports           <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures           <- Generated graphics and figures to be used in reporting
|
├── setup_scripts         <- Scripts or tools for setting up the environment
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
├── environment.yml       <- The conda environment file for reproducing the development environment.
│                            
│
│
└── setup.py              <- Make this project pip installable with `pip install -e`(TODO)

```
## Documentation
The documentation for this project is hosted on our wiki. You can find it here (TODO). Note you need to be on the Nexus Office network.

## Setting up the environment for development

### For use or development on a local machine
#### Prerequisites
We are assuming that Git and a code editor(like VS Code,Neovim,etc) are already installed on your machine.
#### Steps
1. Set up GitHub SSH key-based authentication on your target system(optional but recommended). You can find instructions on how to do that [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

2. Clone the repository to your disired location.
3. Install **miniconda** on your system.
   * Download the installer from [here](https://docs.conda.io/projects/miniconda/en/latest/index.html)
   * Move the installer to the desired location and run it. Here is an example for Linux:
       ```bash
         bash Miniconda3-latest-Linux-x86_64.sh
       ```
   * Follow the instructions on the screen to complete the installation.
   * Restart your terminal.
   * You can now use conda. You can find more information on how to use conda [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
4. By default conda is extrimily slow when working with large environments.That is why you should switch to libmamba solver  to speed up your conda. You can do this by running the following command:
   ```bash
      conda update -n base conda  # might not be required
      conda install -n base conda-libmamba-solver
      conda config --set solver libmamba
   ```
5. Create a conda environment for the project. You can do this by running the following command in the root directory of the project:
   ```bash
     conda env create -f environment.yml
   ```
6. Activate the environment by running the following command:
   ```bash
     conda activate colibri
   ```
   Note: You you made any mistake when working with conda you can simply delete the miniconda directory and start over without any issues.
7. Open the project in your code editor. For VS CODE you can do this by running the following command in the root directory of the project:
   ```bash
     code .
   ```
8. You can run any python file within the project either directly in the code editor or by running the following command in the root directory of the project:
   ```bash
     python3 <path_to_file>
   ```
   For example:
   ```bash
     python3 src/colibri/main.py
   ```
9. You can run any Jupyter notebook within the project by running the following command in the root directory of the project:
   ```bash
     jupyter notebook
   ```
   This will open a new tab in your browser with the Jupyter notebook dashboard. You can then navigate to the desired notebook and open it.

10. You are set up and ready to go!

### For development on a remote machine
This is mostly used for training models or doing any kind of heavy computation that requires a GPU. You can also use this method if you want to develop on a remote machine for any other reason.
#### Prerequisites
We are assuming that you will be using Lena (gs-cs-001 or any other gpu server in our lab) for development. We also assume that the server runs Ubuntu and that it has evrething set up for you. If not you should contact the person responsible for the server.

You should also need the following:
* You have setup a VPN connection. You can find instructions on how to do that [here](https://paper.dropbox.com/doc/How-to-setup-Equal1-Fortinet-Client--B_3JIq0SPcqwscXSEbmmsm6eAg-cp4FPGXVxaElmSLzV7AsS).
* We also recomand you setup the internal SSL certificate. Find out [here](https://paper.dropbox.com/doc/Equal1-Internal-SSL-Certificate-Installation--B_2ojQ0nh36k2CwySfYSX22bAg-QGnhXBSrxdVQI7NcgqnOy).

#### Steps
Most of the steps are taken from [here](https://paper.dropbox.com/doc/How-to-log-into-and-use-Lena--B_22Xp53VqlWqhtOIOi2O9WKAg-j448SnYljlavJzI4EL4Im). If you are stuck at any point or have a different setup you can find more information there. You might not need to do all the steps from the link above.

1. Establish a VPN connection with FortiClient to the Nexus Office network(see prerequisites).
2. Open MobaXterm, click on "Session" in the top left corner, and then click on "SSH".Enter for the remote host the following: **gs-sc-001** and for the username your **Equel1 username**(ex:rbals). Then click on "OK".After that you will be prompted for your **Equel1 password**. Enter it and click on "OK".
3. You are now logged in to Lena. You need to navigate to `/local/`(to understand why check the [link](https://paper.dropbox.com/doc/How-to-log-into-and-use-Lena--B_22Xp53VqlWqhtOIOi2O9WKAg-j448SnYljlavJzI4EL4Im#:uid=581416797056383022945682&h2=Lena-usage-tips)) and create a new folder with you username.You will spend most of your time developoing there. You can do this by running the following commands:
   ```bash
     cd /local/
     mkdir <your_username>
     cd <your_username>
   ```
4. Follow the steps from the previous section to clone the repository and set up the environment(Steps 2-6).
5. Now we need a way to connect with a code editor .For VS CODE follow the instructions from this [page](https://code.visualstudio.com/docs/remote/ssh), but only the subsection called "Installation" and "Connect to a remote host".
6. You should setup your ssh key authtification so you don't need to enter your password every time you connect to Lena. You can find a very good article [here](https://adamtheautomator.com/add-ssh-key-to-vs-code/)(Follow the article until the section "Associating the Public Key with the root User").
7. You can now only use VS CODE if you like, but i also like to still use MobaXterm for some things. You can set a new connection to Lena in MobaXterm by doing what we previsly did with the only diference that you can click on "Advanced SSH settings" and 
click on "Use private key" and select the private key you generated in the previous step.Now you can connect to Lena with only one click and without entering your password.
8. You can add the following line to your `.bashrc` file for easy access to the project:
   ```bash
     # Custom aliases
      alias cdl='cd /local/rbals'
      alias ac="conda activate colibri"
      alias a="conda activate"
      alias d="conda deactivate"

      # Change to work folder after ssh conncection
      cd /local/<your_username>

      # Make sure to add the following line after the conda init calls
      # Activate automatically the conda env
      conda activate colibri

   ```
9. Now you can follow the steps from the previous section to run the project(Steps 7-10).

10. Happy coding!

### Traking large files with Git LFS
Some rules of thumb for when a file should be tracked with Git LFS:
-> big binaries
-> diferent extensions for pictures, plots, etc
-> json, yaml or other text files are not binary but if it doesn't have newlines or it's too big then it's not worth storing as text, and should be stored as binary instead(like the .db files or .hdf5 files)
-> everything that doesn't change compatibly with Diff
-> an exception is made for small binary files of several tens of kb even mb(rarely) or for files that are not changed often

To track a file with Git LFS you need to run the following command:
```bash
  git lfs track <path_to_file>
```

If you add a new type of large file to the project you can track all the files of that type by running the following command:
```bash
  git lfs track "*.<file_extension>"
```

### Repository development guidelines
The point of this section is to provide some guidelines for the development of the repository. These are not strict rules and you can deviate from them if you have a good reason to do so. The goal is to try to set some rules that will make the development process easier for everyone.

#### Code style
The point of a coding style is to make the code more readable and easier to understand.That's where linting tools come into the picture, they help us enforce a said coding style. Please apply a linting tool before committing python code. 
Right now we use **pylint** and **Black** but we might transition to **Ruff** later on.

#### Git workflow
You can find the general workflow and also a good introduction to Git [here](https://github.com/equal1/eq1x-scripts#git-101).
#### Development process
* For now we don't have any restrictions on the develoment process but they might be added in the future.
### References
You can find links to all the papers and articles on the subject of autotuning in the references folder.

You can also see links to publicly available data sets in the data/external folder.

---

Note: Anyone who contributes to this repo can expand this README.md file with any information that they deem useful for the users or other developers.
