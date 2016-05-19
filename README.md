# Introduction
This package contains the OLSRNN implementation for the task of opinion targets recognition. Nevertheless，since the dimensionality of the networks can be changed to match that of the data, it could in principle be used for almost any
supervised sequence labelling task.    
Most of the codes were written in C++:
- bin: some makefiles
- config: well trained models
- include: header files
- lib: external library files
- res: resource files
- src: code files

# Requirements
1. To run the model the open source tools Eigen (a C++ template library for linear algebra) need to be downloaded: http://eigen.tuxfamily.org/index.php?title=Main_Page. Put the folder ```Eigen``` in ```/lib/Eigen```.   
2. If you want to run the model on the  task of opinion targets recognition, you should download:    
Word Embeddings: http://nlp.stanford.edu/data/glove.42B.300d.zip. Put the word embedding file in the folder ```/res```.   
Datasets from SemEval-2014: http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools. Some sample training files have been put in ```/res```, you should preprocess the raw datasets to the same format with the sample files.

# Demo
After successfully completing the [Requirements](#requirements), you'll be ready to run the demo. First of all, you shuold 
We have already trained the model in the filed of Restaurant and Laptop, and the modle files are stored in the the folder ```/comfig```.
