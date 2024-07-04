# Sentiment-Classifier

The goal of this project is to apply machine learning models on social media media posts, focusing on classifying sentiments and mapping polarities to support decision making.
## Instructions

### Requiriments
- ```Conda 24.3.0```   
- ```Python 3.11.7```

### Cloning a GitHub Repository

1. On GitHub.com, navigate to the main page of the repository

2. Above the list of files, click on Code
3. Copy the repository URL:
To clone using HTTPS, click on HTTPS and copy the URL.
To clone using an SSH key, click on SSH and copy the URL.
4. Open the Terminal or Git Bash on your computer.
5. Change the current working directory to where you want the cloned directory to be located.
6. Type the following command and paste the copied URL:    
```bash
   git clone <REPOSITORY_URL>
```
For example:
```bash
   [git clone https://github.com/elir0d/sentiment-classifier.git
```
7. Press ENTER to create the local clone.
### Creating a Conda Environment from a YAML File

1. After cloning the repository, navigate to the directory where the repository was cloned.
2. Inside that directory, you should find a YAML file named environment.yml.
3. Execute the following command to create the Conda environment based on the YAML file:
```bash
conda env create -f environment.yml
```
This will create an environment with the specified dependencies from the YAML file.
Now you have the cloned repository and the Conda environment set up! 🚀 Remember to activate the environment before using it:
- To activate the environment:
```bash
conda activate environment-name
```
- To deactivate the environment:
```bash
conda deactivate
```

## DVC Explanation

**NOTE**: if you won't use DVC, skip this instructions.

This project structure is as an example of how to work with DVC from inside a Jupyter Notebook.

This workflow should enable you to enjoy the full benefits of working with Jupyter Notebooks, while getting most of the benefit out of DVC - 
namely, **reproducible and versioned data science**.


The idea is to leverage DVC in order to create immutable snapshots of your data and models as part of your git commits.
To enable this, you can created the following DVC stages:
1. **Raw data** - kept in `data/raw/`, versioned in `data/raw.dvc` 
2. **Processed data** - kept in `data/processed/`, versioned in `process_data.dvc` 
3. **Trained models** - kept in `models/`, versioned in `models.dvc` 
4. **Metrics** - kept in `metrics/metrics.json`, versioned as part of the git commit and referenced in `models.dvc`

Unlike a typical DVC project, which requires you to refactor your code into modules which are runnable from the command line,
In this project the aim is to enable you to stay in your comfortable notebook home territory.

So, instead of using `dvc repro` or `dvc run` commands, **just run your code as you normally would in [`Example.ipynb`](/Example.ipynb)**. 
We prepared special cells (marked with green headers) inside this notebook that let you run `dvc commit` commands on the relevant
DVC stages defined above, immediately after you create the relevant data files from your notebook code.

[`dvc commit`](https://dvc.org/doc/commands-reference/commit) computes the hash of the versioned data and saves that hash
as text inside the relevant `.dvc` file. The data itself is ignored and not versioned by git, instead being versioned with DVC.
However, the `.dvc` files, being plain text files, ARE checked into git.

So, to summarize, this workflow should enable you to create a git commit which contains all relevant code, together with
*references* to the relevant data and the resulting models and metrics. **Painless reproducible data science!**

It's intended as a guideline - definitely feel free to play around with its structure to suit your own needs.

## Project Structure
```
├── assets.                      <- All projects assets like images, logos, etc...
|
├── data                         <- The original, immutable data dump
|
├── metrics                      <- Relevant metrics after evaluating the model
|
├── utils                        <- This folder serves as a toolbox containing essential tools and helperfunctions 
|                                   specifically designed to automate various machine learning processes. 
|                                   These methods can be reused across different parts of your project.
|
├── envirioment.yml              <- The requirements file for reproducing the analysis environment, e.g.
│                                   generated with `conda env export  > envirioment.yml`
│
├── LICENSE                      <- The project license
├── README.md                    <- The top-level README for developers using this project.
├── Example.ipynb                <- A prototype project notebook that explain how to use DVC
├── Sentiment-Classifier.ipynb   <- The main project notebook
```
---
