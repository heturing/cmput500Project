# cmput500Project
This is the repo for cmput 500 course project

## Introduction
This repository contains the code for CMPUT 500 course project. We also include the dataset used in this project in the `data/LiveVariableAnalysisOut.txt`. The dataset we provide contains 13629 training examples. Each training example is a triple (jimple expression, input data facts, output data facts). All the training examples are generated by Soot analysis framework running Live Variable Analysis. 

## Generating your Dataset
**We do include the dataset used in this project. So, you can run the whole experiment without generating your dataset**.


To generate your own dataset, you need to:
* Compile your all your .java file into .class file. (Soot's frontend is quite outdated and only works with java7)
* Implement and compile your intra-procedural analysis in [Soot](https://github.com/soot-oss/soot). Your analysis problem should output each line of jimple code along with the input data facts and output data facts.
* Change line 44 of extract_ground_truth.py with your compiled analysis.
* Run `python extract_ground_truth.py -pp PATH_TO_YOUR_.CLASS_FILEs`



## Running the code
### Create conda environment
* Run the following command to install conda environment
`conda create --name NAME_OF_YOUR_ENVIRONMENT python=3.7`
* Activate environment
`conda activate NAME_OF_YOUR_ENVIRONMENT`
* Deactivate environment
`conda deactivate`

### Install Dependencies
Run this command to install the dependency libraries `pip install numpy gensim tensorflow-gpu`. Run this `pip install numpy gensim tensorflow` if you want to train the network with CPU.


### Run the experiment

The program will run for several minutes. If you want to train the network with gpu, be sure to install CUDA and Cudnn correctly according to you tensorflow version. The correspondence can be found [here](https://www.tensorflow.org/install/source#gpu). To install CUDA in conda, run this `conda install cudatoolkit=VERSION`. See here for [installing Cudnn](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-linux).

```
cd cmput500Project
python run_experiment.py -nn FNN -df data/LiveVariableAnalysisOut.txt
```

###Options:
* -nn: Specify which neural network to use. (One of FNN, CNN, RNN)
* -df: The path to the dataset file.
* -we: Which word embedding method to use. (One of Word2Vec, Doc2Vec)


