#ifndef UTILITY_TOOL_H
#define UTILITY_TOOL_H

#include<iostream>
#include<vector>
#include"../lib/Eigen/Dense"

using namespace Eigen;
using namespace std;

class Utility
{

public:
    
    static double sigmoid(double);    //sigmoid function
    static double tanhX(double);    //tanh function 
    static void sigmoidforMatrix(MatrixXd&);    //sigmoid function for matrix (coefficient-wise)
    static void tanhXforMatrix(MatrixXd&);    //tanh function for matrix (coefficient-wise)
    static void calculateSoftMax(MatrixXd&);    //calculate softmax values for a vector
    static void disturbOrder(vector<MatrixXd*>&, vector<MatrixXd*>&);    //randomising the order of the geiven sequences
    static void randomInitialize(MatrixXd&, double);    //random Initialize geiven matrix
    static void reverseMarix(MatrixXd&);    //reverse the matrix in the row direction

};

#endif
