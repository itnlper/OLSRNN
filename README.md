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

To Clone this repository, make sure you are working on a Linux system and input:
```
git clone https://github.com/itnlper/OLSRNN.git
```

# Requirements
1. To run the model, the open source tools Eigen (a C++ template library for linear algebra) need to be downloaded: http://eigen.tuxfamily.org/index.php?title=Main_Page. Put the folder ```Eigen``` in ```/lib/Eigen```.   
2. If you want to run the model on the  task of opinion target recognition, download Word Embeddings at: http://nlp.stanford.edu/data/glove.42B.300d.zip, and put the word embedding file in the folder ```/res```.   


# Demo
After successfully completing the [Requirements](#requirements), you'll be ready to run the demo which returns all the opinion targets in the input sentence. We have already trained the model in the filed of Restaurant and Laptop, and the modle files are stored in the the folder ```/comfig```. All the command should be inputed in the folder ```/bin```.    
First of all, you shuold create index for the word embeddings to speed up the process:      
```
make create_index file_name=glove.42B.300d.txt
```
Note that ```glove.42B.300d.txt``` is the sample name of word embedding file, you shuold change it according to the actual situation.    
Then you can run the demo in the domain of Restaurant:  
```
make restaurant
```  
and in the domain of Laptop:
```
make laptop
```  

#Beyond the demo
##Opinion target recognition
If you want to train the model on the task of opinion target recognition,firstly you should download the datasets from SemEval-2014: http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools, and preprocess the raw datasets to the same format with the sample files in the folder ```/res/training```. And then you can trained the model by:  
```
make training_restaurant learning_rate=0.001 momentum_rate=0.9 regularization=2 interation_num=10
```
and
```
make training_laptop learning_rate=0.001 momentum_rate=0.9 regularization=2 interation_num=10
```
You can change the parameters according to the actual situation. To learn more about the parameters, please see the source code. After the training the model will be stored automatically in the folder ```/config``` with a name according to the parameters.
And then you can test the model on the test sets:   
```
make test_restaurant model_file=model_restaurant
```
and   
```
make test_laptop model_file=model_laptop
```
Note that you should change the model_flie to your own.  

##Other Task   
If you want to employ the model to other task, you shuold write a cpp file by yourself. Some reminders are listed here:   
- dsds
- Model initialization: 
```
OLSRNN(int input_cell_num, int memory_cell_num, int output_cell_num)
```
- Model training: 
```
stochasticGradientDescent(vector<MatrixXd*> input_matrix, vector<MatrixXd*> label_matrix, double learning_rate, double momentum_rate, int regularization, int iteration)
```
- Model prediction:
```
vector<MatrixXd*> predict(vector<MatrixXd*> input_matrix)
```

More details can be found in the source code.
