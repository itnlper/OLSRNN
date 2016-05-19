#ifndef OLSRNN_H
#define OLSRNN_H

#include<iostream>
#include<vector>
#include"../lib/Eigen/Dense"

using namespace std;
using namespace Eigen;

class OLSRNN
{

private:

    int input_cell_num;    //cell number of input layer(I)
    int memory_cell_num;    //cell number of memory units layer(H)
    int output_cell_num;    //cell number of output layer(K)

    //weights matrixes
    MatrixXd memory_update_weights;    //the weight matrix between memory layer and update gate (H * H+1)
    MatrixXd input_update_weights;    //the weight matrix between input layer and update gate (H * I+1)
    MatrixXd memory_tanh_weights;    //the weight matrix betweent memory layer and input_tanh layer (H * H+1)
    MatrixXd input_tanh_weights;    //the weight matrix betweent input layer and input_tanh layer (H * I+1)
    MatrixXd memory_output_layer;    //the weight matrix of output softmax layer for hidden_cell (K * H+1)
    MatrixXd output_output_weights;    //the weight matrix between output layers (K * K)

    //value matrixes
    MatrixXd memory_values;    //all memory units layer values at time step 1...T (H * T)
    MatrixXd input_values;    //the sequence of inputs (I * T)

    MatrixXd update_gate_value;    //the values of update gate activation at time step 1...T (H * T)
    MatrixXd inputtanh_value;    //the values of input_tanh layer activation at time step 1...T (H * T)
    MatrixXd output_values;    //the values of output layer for time 1...T (K * T)

    //error matrixes
    MatrixXd error_output;    //the error of output layer (K * T)
    MatrixXd error_memory;    //the error of memory units layer (H * T)
    MatrixXd error_inputtanh;    //the error of input_tanh layer (H * T)
    MatrixXd error_update_gate;    //the error of update gate (H * T)

public:

    OLSRNN();    //default initialization
    OLSRNN(string model_file);    //initialization with the model stored in file
    OLSRNN(int input_cell_num, int memory_cell_num, int output_cell_num);    //random initialization
    void forwardPass(MatrixXd);    //forward psss geiven the input data (matrix)
    void calculateSoftMax(MatrixXd);    //calculate SoftMax for the output layer
    void backwardPass(MatrixXd&);    //calculate errors back through the network
    double calculateError(MatrixXd&);    //calculate the value of cost function
    void checkGradient();    //check the derivative is calculated correct or not
    void updateWeights(
                       MatrixXd &now_input_maxtrix, MatrixXd &now_label_maxtrix,
                       double learning_rate, double momentum_rate,
                       MatrixXd &memory_tanh_weights_derivative,
                       MatrixXd &input_tanh_weights_derivative,
                       MatrixXd &memory_update_weights_derivative,
                       MatrixXd &input_update_weights_derivative,
                       MatrixXd &memory_output_layer_derivative,
                       MatrixXd &output_output_weights_derivative,
                       int regularization = 0
                      );    //update network weights
    void stochasticGradientDescent(vector<MatrixXd*>, vector<MatrixXd*>, double, double, int regularization = 0, int iteration = 1);    //SGD with a momentum
    vector<MatrixXd*> predict(vector<MatrixXd*>);    //predict the label of the geiven input datas
    void saveModel(string file_name);    //save current model into a file
    void loadModel(string file_name);    //load the model from geiven file

    //some getter function
    MatrixXd getOutputValue();    //get the matrix of output_value
    int getInputCellNum();    //get the value of input_cell_num
    int getMemoryCellNum();    //get the value of memory_cell_num
    int getOutputCellNum();    //get the value of output_cell_num
};

#endif
