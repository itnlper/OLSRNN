#include"../include/aspect_tool.h"
#include"../include/utility.h"
#include"../include/string_tool.h"
#include"../include/word2vec_tool.h"
#include"../include/olsrnn.h"
#include<stdexcept>

int main(int argc, char *argv[])
{
    if (argc <= 1)
    {
        throw runtime_error("Missing parameter!");
        return 0;
    }
    
    string function = string(argv[1]);
   
    if (function == "create_index")
    {
        if (argc <= 2)
        {
            throw runtime_error("Missing parameter!");
            return 0;
        }
        string wordvec_file = string(argv[2]);
        Word2VecTool wordvec_tool("../res/" + wordvec_file, false);
        cout << "Creating Index...." << endl;
        wordvec_tool.initWordIndexDict();
        wordvec_tool.saveWordDict("../res/Word_Vector_Index");
        cout << "Complete!" << endl;
    }
    else if (function == "demo_restaurant")
    {
        AspectTool asptool;
        asptool.realTest("../config/model_restaurant");
    }
    else if (function == "demo_laptop")
    {
        AspectTool asptool;
        asptool.realTest("../config/model_laptop");
    }
    else if (function == "training_restaurant" || function == "training_laptop")
    {
        if (argc <= 5)
        {
            throw runtime_error("Missing parameter!");
            return 0;
        }
        AspectTool asp_tool;
        vector<MatrixXd*> input_datas;
        vector<MatrixXd*> input_labels;
        OLSRNN olsrnn(300, 100, 3);

        if (function == "training_restaurant")
            asp_tool.getInputData("../res/restaurant_training", input_datas, input_labels);
        else
            asp_tool.getInputData("../res/laptop_training", input_datas, input_labels);
        //learning rate, momentum rate, regularization, iteration number
        double learning_rate = atof(argv[2]);
        double momentum_rate = atof(argv[3]);
        int regularization = atoi(argv[4]);
        int iteration_num = atoi(argv[5]);
        olsrnn.stochasticGradientDescent(input_datas, input_labels, learning_rate, momentum_rate, regularization, iteration_num);
        ostringstream o_string;
        o_string << "../config/" << function << '_' << learning_rate << '_' << momentum_rate << '_' << regularization << '_' << iteration_num;
        string model_file = o_string.str();
        cout << model_file << endl;
        olsrnn.saveModel(model_file);
    }
    else if (function == "test_restaurant" || function == "test_laptop")
    {
        if (argc <= 2)
        {
            throw runtime_error("Missing parameter!");
            return 0;
        }
        string model_file = string(argv[2]);

        AspectTool asp_tool;
        vector<MatrixXd*> test_datas;
        vector<MatrixXd*> test_labels;
        if (function == "test_restaurant")
            asp_tool.getInputData("../res/restaurant_test", test_datas, test_labels);
        else
            asp_tool.getInputData("../res/laptop_test", test_datas, test_labels);

        OLSRNN olsrnn("../config/" + model_file);
        vector<MatrixXd*> predict_labels = olsrnn.predict(test_datas);
        asp_tool.calF1Score(predict_labels, test_labels);
    }
    return 0;
}
