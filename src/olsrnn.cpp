#include<iostream>
#include<cmath>
#include<fstream>

#include"../include/olsrnn.h"
#include"../include/utility.h"

using namespace std;

//default initialization
OLSRNN::OLSRNN()
{
    return;
}

//initialization with model stored in file
OLSRNN::OLSRNN(string model_file)
{
    loadModel(model_file);
    return;
}


//initialization with layers cell number
OLSRNN::OLSRNN(int input_cell_num, int memory_cell_num, int output_cell_num) : 

    memory_tanh_weights(memory_cell_num, memory_cell_num + 1),
    input_tanh_weights(memory_cell_num, input_cell_num + 1),
    memory_update_weights(memory_cell_num, memory_cell_num + 1),
    input_update_weights(memory_cell_num, input_cell_num + 1),
    memory_output_layer(output_cell_num, memory_cell_num + 1),
    output_output_weights(output_cell_num, output_cell_num)

{
    this -> input_cell_num = input_cell_num;
    this -> memory_cell_num = memory_cell_num;
    this -> output_cell_num = output_cell_num;
    
    /*
    //initialize weights randomly from [-0.1, 0.1], Pseudo Random
    memory_tanh_weights = MatrixXd::Random(memory_cell_num + 1, 1) * 0.1;
    input_tanh_weights = MatrixXd::Random(input_cell_num + 1, 1) * 0.1;
    memory_update_weights = MatrixXd::Random(memory_cell_num + 1, 1) * 0.1;
    input_update_weights = MatrixXd::Random(input_cell_num + 1, 1) * 0.1;
    memory_output_layer = MatrixXd::Random(output_cell_num, memory_cell_num + 1) * 0.1;
    output_output_weights = MatrixXd::Random(output_cell_num, output_cell_num) * 0.1;
    */

    //initialize weigths randomly with gieven precision, True Random
    Utility::randomInitialize(memory_tanh_weights, 0.1);
    Utility::randomInitialize(input_tanh_weights, 0.1);
    Utility::randomInitialize(memory_update_weights, 0.1);
    Utility::randomInitialize(input_update_weights, 0.1);
    Utility::randomInitialize(memory_output_layer, 0.1);
    Utility::randomInitialize(output_output_weights, 0.1);

    return;
}

//forward pass geiven the input data (matrix)
void OLSRNN::forwardPass(MatrixXd temp_input_matrix)
{
    /*
     * input_matrix : input sequence (without bias) (I * T)
     */
    
    int sequence_length = temp_input_matrix.cols();    //sequence length

    //initialize memory matrix with zero
    memory_values = MatrixXd::Zero(memory_cell_num, sequence_length);
    inputtanh_value = MatrixXd::Zero(memory_cell_num, sequence_length);
    update_gate_value = MatrixXd::Zero(memory_cell_num, sequence_length);
    output_values = MatrixXd::Zero(output_cell_num, sequence_length);

    //add bias for input data
    int input_row_num = temp_input_matrix.rows();
    int input_col_num = temp_input_matrix.cols();
    MatrixXd input_matrix(input_row_num + 1, input_col_num);
    input_matrix.block(0, 0, input_row_num, input_col_num) = temp_input_matrix;
    input_matrix.block(input_row_num, 0, 1, input_col_num) = MatrixXd::Ones(1, input_col_num);

    //for every input, calculate correlative values of inner cells
    for(int col_index = 0; col_index < sequence_length; ++col_index)
    {
    	const MatrixXd &temp_input = input_matrix.col(col_index);
    	const MatrixXd temp_input_trans = temp_input.transpose();
    	MatrixXd temp_hidden_trans(1, memory_cell_num + 1);
    	if (col_index == 0)
    	{
    		temp_hidden_trans.block(0, 0, 1, memory_cell_num) = MatrixXd::Zero(1, memory_cell_num);    //initialization with all zero
    	}
    	else
    	{
    		temp_hidden_trans.block(0, 0, 1, memory_cell_num) = memory_values.col(col_index - 1).transpose();    //previous memory cells value
    	}
    	temp_hidden_trans(0, memory_cell_num) = 1;    //add bias term
    	
    	MatrixXd temp_hidden_result;    //Temporary matrix
    	MatrixXd temp_input_result;    //Temporary matrix

    	//calculate the activation of update gate
    	temp_hidden_result = memory_update_weights * temp_hidden_trans.transpose();
    	temp_input_result = input_update_weights * temp_input;
        MatrixXd temp_update_gate_value = temp_hidden_result + temp_input_result;
        Utility::sigmoidforMatrix(temp_update_gate_value);
    	update_gate_value.col(col_index) = temp_update_gate_value;
    	
    	//calculate the activation of update gate
    	temp_hidden_result = memory_tanh_weights * temp_hidden_trans.transpose();
    	temp_input_result = input_tanh_weights * temp_input;
        MatrixXd temp_inputtanh_value = temp_hidden_result + temp_input_result;
        Utility::tanhXforMatrix(temp_inputtanh_value);
    	inputtanh_value.col(col_index) = temp_inputtanh_value;
        
        //calculate new memory cells values
        MatrixXd new_hidden_result = temp_hidden_trans.transpose().block(0, 0, memory_cell_num, 1).array() * temp_update_gate_value.array() + temp_inputtanh_value.array() * (1 - temp_update_gate_value.array());
        memory_values.col(col_index) = new_hidden_result;

        //calculate output layer values(softmax layer)
        MatrixXd new_hidden_bias(memory_cell_num + 1, 1);
        new_hidden_bias.block(0, 0, memory_cell_num, 1) = new_hidden_result;
        new_hidden_bias(memory_cell_num, 0) = 1;    //add bias
        MatrixXd temp_output_val = memory_output_layer * new_hidden_bias;    //(k, h + 1) * (h + 1, 1) = (k, 1)
        if (col_index > 0)
        {
            temp_output_val += output_output_weights * output_values.col(col_index - 1);
        }
        Utility::calculateSoftMax(temp_output_val);
        output_values.col(col_index) = temp_output_val;

    }//end 'for'
    return;
}//end 'forwardPass'

//calculate SoftMax for the output layer
void OLSRNN::calculateSoftMax(MatrixXd output_values)
{
    /*
     * note : this function should be invoked after forwardPass()
     * argument : output_values -- the output layer values needed to calculate SoftMax (K * T)
     * function : calculate SoftMax for the output layer
     */
    for (int t = 0; t < output_values.cols(); ++t)
    {
        double sum = 0;
        for (int row_num = 0; row_num < output_values.rows(); ++row_num)
        {
            output_values(row_num, t) = exp(output_values(row_num, t));
            sum += output_values(row_num, t);
        }
        for (int row_num = 0; row_num < output_values.rows(); ++row_num)
        {
            output_values(row_num, t) = output_values(row_num, t) / sum;
        }
    }
    
    this -> output_values = output_values;
    return;
}

//calculate errors back through the network
void OLSRNN::backwardPass(MatrixXd &label)
{
    /*
     * note : this function should be invoked after forwardPass()
     * argument : label -- the labels of input datas (K * T)
     * function : calculate errors back through the network
     */
    
    int sequence_length = label.cols();    //sequence length

    //initialize memory matrixes
    error_output = MatrixXd::Zero(output_cell_num, sequence_length);
    error_memory = MatrixXd::Zero(memory_cell_num, sequence_length);
    error_inputtanh = MatrixXd::Zero(memory_cell_num, sequence_length);
    error_update_gate = MatrixXd::Zero(memory_cell_num, sequence_length);

    for (int col_num = sequence_length - 1; col_num >= 0; --col_num)
    {
        //calculate the error of output layer (softmax layer)
        //Part 1 : calculate errors from current output
        for (int row_num = 0; row_num < output_cell_num; ++row_num)
        {
            if (label(row_num, col_num) == 1)
            {
                error_output(row_num, col_num) = output_values(row_num, col_num) - 1;
            }
            else
            {
                error_output(row_num, col_num) = output_values(row_num, col_num);
            }
        }
        //Part 2 : calculate errors from next time-step output layer
        if (col_num != sequence_length - 1)
        {
            MatrixXd temp_error = output_output_weights.transpose() * error_output.col(col_num + 1);
            double temp_sum = (output_values.col(col_num).array() * temp_error.array()).sum();
            error_output.col(col_num).array() += output_values.col(col_num).array() * (temp_error.array() - temp_sum);
        }

        /*
         * calculate memory units errors
         */ 
        //Part 1 : calculate errors from current output layer
        error_memory.col(col_num) = (memory_output_layer.transpose() * error_output.col(col_num)).block(0, 0, memory_cell_num, 1);
        if (col_num != sequence_length - 1)
        {
            //Part 2 : calculate errors from next input_tanh layer 
            error_memory.col(col_num) += (memory_tanh_weights.transpose() * error_inputtanh.col(col_num + 1)).block(0, 0, memory_cell_num, 1);
            //Part 3 : calculate errors from next update gate 
            error_memory.col(col_num) += (memory_update_weights.transpose() * error_update_gate.col(col_num + 1)).block(0, 0, memory_cell_num, 1);
            //Part 4 : calculate errors from next memory units layer
            error_memory.col(col_num).array() += error_memory.col(col_num + 1).array() * update_gate_value.col(col_num + 1).array();
        }

        //calculate update gate errors
        error_update_gate.col(col_num) = error_memory.col(col_num).array() * (0 - inputtanh_value.col(col_num).array());
        if (col_num > 0)
        {
            error_update_gate.col(col_num).array() += memory_values.col(col_num - 1).array() * error_memory.col(col_num).array();
        }
        error_update_gate.col(col_num).array() *= update_gate_value.col(col_num).array() * (1 - update_gate_value.col(col_num).array());

        //calculate input_tanh layer errors 
        error_inputtanh.col(col_num) = (1 - inputtanh_value.col(col_num).array() * inputtanh_value.col(col_num).array()) *
                                         error_memory.col(col_num).array() * (1 - update_gate_value.col(col_num).array());

    }//end 'for'
    return;
}//end 'backwardPass()'

//calculate the value of cost function
double OLSRNN::calculateError(MatrixXd &label)
{
    /*
     * note : this function should be invoked after 'forwardPass()'
     * argument : label : the label of input data (K * T)
     * return : the error of geiven input data
     */

    double error = 0;
    for (int col_num = 0; col_num < label.cols(); ++col_num)
    {
        for (int row_num = 0; row_num < label.rows(); ++row_num)
        {
            if (label(row_num, col_num) == 1)
            {
                error += -log(output_values(row_num, col_num));
                break;
            }
        }
    }
    return error;
}//end 'calculateError()'

//check the derivative is calculated correct or not
void OLSRNN::checkGradient()
{
    /*
     * function : check the derivative of oupout tanh layer's weights
     * note : this function is not integrate but effective
     */

    double epsilon = 1e-4;    //EPSILON

    //initialize a random input data and label
    int sequence_length = 2;
    MatrixXd input_data = MatrixXd::Random(input_cell_num, sequence_length);
    MatrixXd label = MatrixXd::Zero(output_cell_num, sequence_length);
    srand((unsigned int)time(NULL));
    for (int col_num = 0; col_num < sequence_length; ++col_num)
    {
        int true_row = rand() % output_cell_num;
        label(true_row, col_num) = 1;
    }

    //calculate derivative with the model
    forwardPass(input_data);
    backwardPass(label);
    //double cal_value = (state_values.row(0) * error_output_tanh.row(0).transpose())(0, 0);
    double cal_value = (error_inputtanh.row(0) * input_data.row(0).transpose())(0, 0);

    //calculate derivative with approximation
    double sim_value = 0;
    input_tanh_weights(0, 0) += epsilon;
    forwardPass(input_data);
    sim_value = calculateError(label);
    input_tanh_weights(0, 0) -= 2 * epsilon;
    forwardPass(input_data);
    sim_value -= calculateError(label);
    sim_value /= 2 * epsilon;
    
    cout << "cal_value  " << cal_value << endl;
    cout << "sim_value  " << sim_value << endl;

    //restore the real weight
    input_tanh_weights(0, 0) += epsilon;
    
    return;
}

//update network weights
void OLSRNN::updateWeights(
                           MatrixXd &now_input_maxtrix, MatrixXd &now_label_maxtrix,
                           double learning_rate, double momentum_rate,
                           MatrixXd &memory_tanh_weights_derivative,
                           MatrixXd &input_tanh_weights_derivative,
                           MatrixXd &memory_update_weights_derivative,
                           MatrixXd &input_update_weights_derivative,
                           MatrixXd &memory_output_layer_derivative,
                           MatrixXd &output_output_weights_derivative,
                           int regularization
                          )
{
    /*
     * argument : now_input_maxtrix -- current input data (I * T)
     *            now_label_maxtrix -- current label data (K * T)
     *            learning_rate -- initial learning rate
     *            momentum_rate -- the parameter of momentum
     *            other matrixes -- record last derivative for momentum
     *            regularization -- regularization mothed with 0 for none, 1 for L1 (to be update), 2 for L2
     */

        int sequence_length = now_input_maxtrix.cols();    //the length of current sequence

        //calculate zhe momentum term
        memory_tanh_weights_derivative *= momentum_rate;
        input_tanh_weights_derivative *= momentum_rate;
        memory_update_weights_derivative *= momentum_rate;
        input_update_weights_derivative *= momentum_rate;
        memory_output_layer_derivative *= momentum_rate;
        output_output_weights_derivative *= momentum_rate;

        //calculate derivative with BPTT
        for (int t = 0; t < sequence_length; ++t)
        {
            //define input and hidden vector
            MatrixXd input_now(input_cell_num + 1, 1);
            MatrixXd hidden_previous(memory_cell_num + 1, 1);
            MatrixXd hidden_now(memory_cell_num + 1, 1);

            //initialize input and hidden vector and add bias for them
            input_now.block(0, 0, input_cell_num, 1) = now_input_maxtrix.col(t);
            input_now(input_cell_num, 0) = 1;
            hidden_now.block(0, 0, memory_cell_num, 1) = memory_values.col(t);
            hidden_now(memory_cell_num, 0) = 1;
            
            if (t > 0)
            {
                hidden_previous.block(0, 0, memory_cell_num, 1) = memory_values.col(t - 1);
            }
            else
            {
                hidden_previous.block(0, 0, memory_cell_num, 1) = MatrixXd::Zero(memory_cell_num, 1);
            }
            
            hidden_previous(memory_cell_num, 0) = 1;

            //update update_gate weights derivative (for input and memory layer)
            memory_update_weights_derivative += error_update_gate.col(t) * hidden_previous.transpose() * learning_rate;
            input_update_weights_derivative +=  error_update_gate.col(t) * input_now.transpose() * learning_rate;

            //update input_tanh layer weights derivative (for input and memory layer)
            memory_tanh_weights_derivative += error_inputtanh.col(t) * hidden_previous.transpose() * learning_rate;
            input_tanh_weights_derivative += error_inputtanh.col(t) * input_now.transpose() * learning_rate;

            //update output layer weights derivative (softmax layer)
            memory_output_layer_derivative += error_output.col(t) * hidden_now.transpose() * learning_rate;
            if (t > 0)
            {
                output_output_weights_derivative += error_output.col(t) * output_values.col(t - 1).transpose() * learning_rate;
            }

        }//end 'for' of sequence

        //regularization
        if (regularization == 1)
        {
            ;
        }
        else if (regularization == 2)
        {
            memory_tanh_weights_derivative += memory_tanh_weights * learning_rate * 0.01;
            input_tanh_weights_derivative += input_tanh_weights * learning_rate * 0.01;
            memory_update_weights_derivative += memory_update_weights * learning_rate * 0.01;
            input_update_weights_derivative += input_update_weights * learning_rate * 0.01;
            memory_output_layer_derivative += memory_output_layer * learning_rate * 0.01;
            output_output_weights_derivative += output_output_weights * learning_rate * 0.01;
        }

        //update weights
        memory_tanh_weights -= memory_tanh_weights_derivative;
        input_tanh_weights -= input_tanh_weights_derivative;
        memory_update_weights -= memory_update_weights_derivative;
        input_update_weights -= input_update_weights_derivative;
        memory_output_layer -= memory_output_layer_derivative;
        output_output_weights -= output_output_weights_derivative;
        
        double ave_error = calculateError(now_label_maxtrix) / sequence_length;
        cout << "Ave_error : " << ave_error << endl;
    
    return;
}


//SGD with a momentum
void OLSRNN::stochasticGradientDescent(vector<MatrixXd*> input_datas, vector<MatrixXd*> input_labels, double learning_rate, double momentum_rate, int regularization, int iteration)
{
    /* 
     * argument : input_datas -- a vector of input data, every element is a I * T matrix
     *            input_labels -- a vector of input labels, every element is a K * T matrix
     *            learning_rate -- initial learning rate
     *            momentum_rate -- the parameter of momentum
     *            regularization -- regularization mothed with 0 for none, 1 for L1 (to be update), 2 for L2
     */
    
    //matrixes record previous derivative
    MatrixXd memory_tanh_weights_derivative = MatrixXd::Zero(memory_cell_num, memory_cell_num + 1);
    MatrixXd input_tanh_weights_derivative = MatrixXd::Zero(memory_cell_num, input_cell_num + 1);
    MatrixXd memory_update_weights_derivative = MatrixXd::Zero(memory_cell_num, memory_cell_num + 1);
    MatrixXd input_update_weights_derivative = MatrixXd::Zero(memory_cell_num, input_cell_num + 1);
    MatrixXd memory_output_layer_derivative = MatrixXd::Zero(output_cell_num, memory_cell_num + 1);
    MatrixXd output_output_weights_derivative = MatrixXd::Zero(output_cell_num, output_cell_num);
    
    //the number of current input
    int input_count = 0;
    //total number of input datas
    int input_num = input_datas.size();

    for (int i = 0; i < iteration; ++i)
    {

    if (i == 5)
    {
        learning_rate *= 0.1;
    }

    //randomising the order of input datas and labels
    Utility::disturbOrder(input_datas, input_labels);

    for (int input_index = 0; input_index < input_num; ++input_index)
    {
        ++input_count;
        
        MatrixXd now_input_maxtrix = *(input_datas[input_index]);    //current input data
        MatrixXd now_label_maxtrix = *(input_labels[input_index]);    //current input label
        
        forwardPass(now_input_maxtrix);
        backwardPass(now_label_maxtrix);

        int sequence_length = now_input_maxtrix.cols();    //the length of current sequence

        //calculate zhe momentum term
        memory_tanh_weights_derivative *= momentum_rate;
        input_tanh_weights_derivative *= momentum_rate;
        memory_update_weights_derivative *= momentum_rate;
        input_update_weights_derivative *= momentum_rate;
        memory_output_layer_derivative *= momentum_rate;
        output_output_weights_derivative *= momentum_rate;

        //calculate derivative with BPTT
        for (int t = 0; t < sequence_length; ++t)
        {
            //define input and memory vector
            MatrixXd input_now(input_cell_num + 1, 1);
            MatrixXd hidden_previous(memory_cell_num + 1, 1);
            MatrixXd hidden_now(memory_cell_num + 1, 1);

            //initialize input and memory vector and add bias for them
            input_now.block(0, 0, input_cell_num, 1) = now_input_maxtrix.col(t);
            input_now(input_cell_num, 0) = 1;
            hidden_now.block(0, 0, memory_cell_num, 1) = memory_values.col(t);
            hidden_now(memory_cell_num, 0) = 1;
            
            if (t > 0)
            {
                hidden_previous.block(0, 0, memory_cell_num, 1) = memory_values.col(t - 1);
            }
            else
            {
                hidden_previous.block(0, 0, memory_cell_num, 1) = MatrixXd::Zero(memory_cell_num, 1);
            }
            hidden_previous(memory_cell_num, 0) = 1;

            //updata update_gate weights derivative (for input and memory layer)
            memory_update_weights_derivative += error_update_gate.col(t) * hidden_previous.transpose() * learning_rate;
            input_update_weights_derivative +=  error_update_gate.col(t) * input_now.transpose() * learning_rate;

            //update input_tanh layer weights derivative (for input and memory layer)
            memory_tanh_weights_derivative += error_inputtanh.col(t) * hidden_previous.transpose() * learning_rate;
            input_tanh_weights_derivative += error_inputtanh.col(t) * input_now.transpose() * learning_rate;

            //updata output layer weights derivative (softmax layer)
            memory_output_layer_derivative += error_output.col(t) * hidden_now.transpose() * learning_rate;
            if (t > 0)
            {
                output_output_weights_derivative += error_output.col(t) * output_values.col(t - 1).transpose() * learning_rate;
            }

        }//end 'for' of sequence

        //regularization
        if (regularization == 1)
        {
            ;
        }
        else if (regularization == 2)
        {
            memory_tanh_weights_derivative += memory_tanh_weights * learning_rate * 0.01;
            input_tanh_weights_derivative += input_tanh_weights * learning_rate * 0.01;
            memory_update_weights_derivative += memory_update_weights * learning_rate * 0.01;
            input_update_weights_derivative += input_update_weights * learning_rate * 0.01;
            memory_output_layer_derivative += memory_output_layer * learning_rate * 0.01;
            output_output_weights_derivative += output_output_weights * learning_rate * 0.01;
        }

        //updata weights
        memory_tanh_weights -= memory_tanh_weights_derivative;
        input_tanh_weights -= input_tanh_weights_derivative;
        memory_update_weights -= memory_update_weights_derivative;
        input_update_weights -= input_update_weights_derivative;
        memory_output_layer -= memory_output_layer_derivative;
        output_output_weights -= output_output_weights_derivative;
        
        double ave_error = calculateError(now_label_maxtrix) / sequence_length;
        cout << ave_error << endl;

    }//end 'for' of input datas
    
    }//end 'for' of pass num
    return;
}

//predict the output of the geiven input datas
vector<MatrixXd*> OLSRNN::predict(vector<MatrixXd*> input_datas)
{
    /*
     * argument : input_datas -- the data waiting to be predict, every element is a matrix pointer (I * T)
     * return : predict_labels -- the label of input datas predicted by LSTM, every element is a matrix pointer (K * T)
     */

    vector<MatrixXd*> predict_labels;    //vector to store the predict labels
    int input_num = input_datas.size();    //the number of input datas

    //for every input predict the label
    for (int i = 0; i < input_num; ++i)
    {
        MatrixXd temp_input_data = *(input_datas[i]);
        int sequence_length = temp_input_data.cols();

        forwardPass(temp_input_data);    //forward pass two calculte the value of every node

        MatrixXd *temp_predict_label = new MatrixXd(output_cell_num, sequence_length);
        *temp_predict_label = MatrixXd::Zero(output_cell_num, sequence_length);

        for (int col_num = 0; col_num < sequence_length; ++col_num)
        {
            int temp_row = 0;
            int temp_col = 0;
            int *p_temp_row = &temp_row;
            int *p_temp_col = &temp_col;
            output_values.col(col_num).maxCoeff(p_temp_row, p_temp_col);
            (*temp_predict_label)(*p_temp_row, col_num) = 1;
        }

        predict_labels.push_back(temp_predict_label);
    }

    return predict_labels;
}

//save the model into a file
void OLSRNN::saveModel(string file_name)
{
    /*
     * argument : file_name -- the file to store the model
     * note : the first line of the file will store various nodes number with the Declaration order.
     *        the weights of model will be then writen into the file with the Declaration order.
     */
    
    ofstream of_model(file_name.c_str());

    char block = ' ';
    of_model << input_cell_num << block << memory_cell_num << block << output_cell_num << endl;
    of_model << memory_tanh_weights << endl;
    of_model << input_tanh_weights << endl;
    of_model << memory_update_weights << endl;
    of_model << input_update_weights << endl;
    of_model << memory_output_layer << endl;
    of_model << output_output_weights << endl;
    
    of_model.close();
    return;
}

//load model from geiven file
void OLSRNN::loadModel(string file_name)
{
    /*
     * argument : file_name -- the file saves the weights of model
     * note : the first line of the file will store various nodes number with the Declaration order.
     *        the weights of model will be then writen into the file with the Declaration order.
     */
    
    ifstream if_model(file_name.c_str());

    //load various layer cell numner with the declaration order
    if_model >> input_cell_num >> memory_cell_num >> output_cell_num;

    //resize the weights matrixes
    memory_tanh_weights.resize(memory_cell_num, memory_cell_num + 1);
    input_tanh_weights.resize(memory_cell_num, input_cell_num + 1);
    memory_update_weights.resize(memory_cell_num, memory_cell_num + 1);
    input_update_weights.resize(memory_cell_num, input_cell_num + 1);
    memory_output_layer.resize(output_cell_num, memory_cell_num + 1);
    output_output_weights.resize(output_cell_num, output_cell_num);
    
    //load memory_tanh_weights
    for (int row_num = 0; row_num < memory_cell_num; ++row_num)
    {
        for (int col_num = 0; col_num < memory_cell_num + 1; ++col_num)
        {
            if_model >> memory_tanh_weights(row_num, col_num);
        }
    }

    //load input_tanh_weights
    for (int row_num = 0; row_num < memory_cell_num; ++row_num)
    {
        for (int col_num = 0; col_num < input_cell_num + 1; ++col_num)
        {
            if_model >> input_tanh_weights(row_num, col_num);
        }
    }

    //load memory_update_weights
    for (int row_num = 0; row_num < memory_cell_num; ++row_num)
    {
        for (int col_num = 0; col_num < memory_cell_num + 1; ++col_num)
        {
            if_model >> memory_update_weights(row_num, col_num);
        }
    }

    //load input_update_weights
    for (int row_num = 0; row_num < memory_cell_num; ++row_num)
    {
        for (int col_num = 0; col_num < input_cell_num + 1; ++col_num)
        {
            if_model >> input_update_weights(row_num, col_num);
        }
    }

    //load memory_output_layer
    for (int row_index = 0; row_index < output_cell_num; ++row_index)
    {
        for (int col_index = 0; col_index < memory_cell_num + 1; ++ col_index)
        {
            if_model >> memory_output_layer(row_index, col_index);
        }
    }

    //load output_output_weights
    for (int row_index = 0; row_index < output_cell_num; ++row_index)
    {
        for (int col_index = 0; col_index < output_cell_num; ++col_index)
        {
            if_model >> output_output_weights(row_index, col_index);
        }
    }

    if_model.close();

    return;
}

//get the matrix of output_values
MatrixXd OLSRNN::getOutputValue()
{
    return output_values;
}

//get the value of input_cell_num
int OLSRNN::getInputCellNum()
{
    return input_cell_num;
}

//get the value of memory_cell_num
int OLSRNN::getMemoryCellNum()
{
    return memory_cell_num;
}

//get the value of output_cell_num
int OLSRNN::getOutputCellNum()
{
    return output_cell_num;
}
