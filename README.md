# cmput500Project
This is the repo for cmput 500 course project

## Introduction
This repository contains the code for CMPUT 500 course project. We also include the dataset used in this project into the data/ folder.

## Generating your Dataset
The dataset contains about 14000 training examples. Each training example is a triple (jimple expression, input data facts, output data facts).
All the training examples are generated by Soot analysis framework running Live Variable Analysis. To gather your own dataset, you can run `python extract_ground_truth.py`.
However, this scripy run the Soot framework. So a prewrite java program must be prepared. You can specify your own analysis in soot by changing the line
44 of extract_ground_truth.py into the command that runs the analysis in Soot.

## Running the code
### Install Dependencies
Run this command to install the dependency libraries `pip install numpy gensim tensorflow-gpu`

### Run the experiment
```
cd cmput500Project
python run_experiment.py 
