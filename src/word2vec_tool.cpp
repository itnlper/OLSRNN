#include"../include/word2vec_tool.h"
#include"../include/string_tool.h"
#include<cstdio>
#include<algorithm>
#include<stdexcept>
#include<fstream>

using namespace Eigen;

//Initialize the object with word vector file path
Word2VecTool::Word2VecTool(const string word2vec_file, bool has_header = false)
{
    FILE *w2v_p = fopen(word2vec_file.c_str(), "rb");

    //check the file exist or not
    if (w2v_p == NULL)
    {
        cout << word2vec_file << " file does not exist!" << endl;
        throw runtime_error("file dose not exist !");
        return;
    }
    
    //skip the header of the file
    if (has_header)
    {
        while (fgetc(w2v_p) != '\n')
        {
            ;
        }
    }

    //skik the word at the head of line
    while (fgetc(w2v_p) != ' ')
    {
        ;
    }

    int vector_length = 0;
    double __wights;
    
    //count vector length
    do 
    {
        fscanf(w2v_p, "%lf", &__wights);
        ++vector_length;
    } while (fgetc(w2v_p) != '\n');

    this -> word2vec_file = word2vec_file;
    this -> has_header = has_header;
    this -> vector_length = vector_length;

    fclose(w2v_p);

    return;
}

//Initialize word_index_dict with geiven word2vec file
void Word2VecTool::initWordIndexDict()
{
    FILE *w2v_p = fopen(word2vec_file.c_str(), "rb");    //open the file with the mode 'binary'

    //check the file exist or not
    if (w2v_p == NULL)
    {
        cout << word2vec_file << " file does not exist!" << endl;
        throw runtime_error("file dose not exist !");
        return;
    }

    char *temp_word = new char[10000];
    double temp_weight;
    long long now_pos = 0;    //the word offset in the word2vec file

    //cut off the header of the word vector file
    if (has_header)
    {
        while (fgetc(w2v_p) != '\n')
        {
            ;
        }
        now_pos += ftell(w2v_p);
    }

    while (true)
    {
        fscanf(w2v_p, "%s", temp_word);    //read the word
        word_index_dict.insert(map<string, long long>::value_type(temp_word, now_pos));     //add the word into word2vec dict
        char flag;
        //if file_pointer meet the end of a line or EOF , break the recurrent
        while ((flag = fgetc(w2v_p)) != '\n' && flag != EOF)
        {
            fscanf(w2v_p, "%lf", &temp_weight);
        }
        if (flag == EOF)
        {
            break;
        }
        else
        {
            now_pos = ftell(w2v_p);    //record the offset
            //print out the word_index_dict
            if (word_index_dict.size() % 10000 == 0)
            {
                cout << word_index_dict.size() << endl; 
            }
        }
    }

    fclose(w2v_p);

    return;
}

//load the word index dictionary from file
void Word2VecTool::loadWordDict(string file_name)
{
    ifstream i_stream(file_name.c_str());
    
    if (!i_stream)
    {
        throw runtime_error(file_name + " opens failed !");
    }

    while (true)
    {
        string word;
        long long word_offset;
        i_stream >> word >> word_offset;
        if (i_stream)
        {
            word_index_dict.insert(map<string, long long>::value_type(word, word_offset));     //add the word into word2vec dict
        }
        else
        {
            break;
        }
    }

    cout << "load dictionary completely" << endl;

    return;
}

//save the word index dictionary into file
void Word2VecTool::saveWordDict(string file_name)
{
    ofstream o_stream(file_name.c_str());
    
    if (!o_stream)
    {
        throw runtime_error(file_name + " opens failed !");
    }

    for (map<string, long long>::iterator mi = word_index_dict.begin(); mi != word_index_dict.end(); ++mi)
    {
        o_stream << mi -> first << ' ' << mi -> second << endl;
    }

    return;
}

//get a word in word vector dictionary which has mimimal edit distence with the geiven word
string Word2VecTool::getMinEditDistenceWord(string word)
{
    /*
     * argument : word -- the geiven word
     * return : the word has mimimal edit distence in the dictionary
     * note : if some words have the same minimal edit distence with the geiven word, we choose the longest one among them,
     *        because we believe the longer a word is, the more infomation it can take.
     */
    
    StringTool st;

    int min_dis = 10000;
    string min_word = "";
    int max_length = 0;
    int word_length = word.size();

    //for every word in the word vector dict, calculate edit distence to find the minimal one
    for (map<string, long long>::iterator map_it = word_index_dict.begin(); map_it != word_index_dict.end(); ++map_it)
    {
        string current_word = map_it -> first;
        int current_word_length = current_word.size();
        if (abs(current_word_length - word_length) <= min_dis)
        {
            int current_dis = st.calEditDistence(current_word, word);
            if (current_dis < min_dis)
            {
                min_dis = current_dis;
                min_word = current_word;
                max_length = current_word_length;
            }
            else if (current_dis == min_dis)
            {
                if (current_word_length > max_length)
                {
                    min_word = current_word;
                    max_length = current_word_length;
                }
            }
        }
    }

    return min_word;
}

void Word2VecTool::getWrodVect(string word, MatrixXd &word_vector)
{
    /*
     * argument : word -- the word
     *            word_vector -- a col vector of the word
     */
        
    transform(word.begin(), word.end(), word.begin(), towlower);    //transform Uper to Lower

    FILE *w2v_p = fopen(word2vec_file.c_str(), "rb");    //open the file with the mode 'binary'

    //check the file exist or not
    if (w2v_p == NULL)
    {
        cout << word2vec_file << " file does not exist!" << endl;
        throw runtime_error("file dose not exist !");
        return;
    }

    //word_vec dict contain the word or not
    if (word_index_dict.find(word) == word_index_dict.end())
    {
        cout << '\'' << word << '\'' << " dose not exist in the word2vector file !" << endl;
        
        //if '-' exists in the word or not
        if (word.find_first_of("-", 0) != string::npos)
        {
            /*
             * note : if '-' exists in the word, take the average word vector of each word splited by "-"
             */

            int valid_word_count = 0;
            MatrixXd word_vector = MatrixXd::Zero(getVectorLength(), 1);

            StringTool st;
            vector<string> word_list;
            st.split(word, "-", word_list);    //split the word with "-"

            //add all valid word vector splited by "-"
            for (int i = 0; i < word_list.size(); ++i)
            {
                string temp_word = word_list[i];
                //assume that the word with more than one char is valid
                if (temp_word.size() > 1)
                {
                    MatrixXd temp_vector = MatrixXd::Zero(getVectorLength(), 1);
                    getWrodVect(temp_word, temp_vector);
                    word_vector += temp_vector;
                    ++valid_word_count;
                }
            }

            if (valid_word_count > 0)
            {
                word_vector /= valid_word_count;
            }

        }
        else if (word.size() >= 2 && (word.rfind("'s") == word.size() - 2 || word.rfind("s'") == word.size() - 2))
        {
            /*
             * if the word ends with 's or s', suppose the original word is the one cut off 's or s' from the word.
             * Then add the original word vector and the word vector of 's, and get the average
             */
            
            //get the original word and the 's postfix
            string word_1 = word.substr(0, word.size() - 2);
            string word_2 = "'s";

            //get the original word's word vector
            MatrixXd temp_vector = MatrixXd::Zero(getVectorLength(), 1);
            getWrodVect(word_1, temp_vector);
            word_vector += temp_vector;
            
            //get the postfix's word vector
            temp_vector = MatrixXd::Zero(getVectorLength(), 1);
            getWrodVect(word_2, temp_vector);
            word_vector += temp_vector;

            //get average
            word_vector /= 2;
        }
        else if (word.find("$") == 0)
        {
            /*
             * if the word starts with $, separate the $ and the other part,
             * Then add the $ vector and the word vector of the other part, and get the average
             */
            
            //get the $ and the other part of the word 
            string word_1 = "$";
            string word_2 = word.substr(1);

            //get the $'s word vector
            MatrixXd temp_vector = MatrixXd::Zero(getVectorLength(), 1);
            getWrodVect(word_1, temp_vector);
            word_vector += temp_vector;
            
            //get the other part's word vector
            temp_vector = MatrixXd::Zero(getVectorLength(), 1);
            getWrodVect(word_2, temp_vector);
            word_vector += temp_vector;

            //get average
            word_vector /= 2;
        }
        else if (word.size() >= 3 && word.rfind("'ll") == word.size() - 3)
        {
            /*
             * if the word ends with 'll, separate the 'll and the other part,
             * Then add the 'll vector and the word vector of the other part, and get the average
             */
            
            //get the 'll and the other part of the word 
            string word_1 = "'ll";
            string word_2 = word.substr(0, word.size() - 3);

            //get the 'll's word vector
            MatrixXd temp_vector = MatrixXd::Zero(getVectorLength(), 1);
            getWrodVect(word_1, temp_vector);
            word_vector += temp_vector;
            
            //get the other part's word vector
            temp_vector = MatrixXd::Zero(getVectorLength(), 1);
            getWrodVect(word_2, temp_vector);
            word_vector += temp_vector;

            //get average
            word_vector /= 2;
        }
        else if (word.size() >= 3 && word.rfind("n't") == word.size() - 3)
        {
            /*
             * if the word ends with n't, separate the n't and the other part,
             * Then add the n't vector and the word vector of the other part, and get the average
             */
            
            //get the n't and the other part of the word 
            string word_1 = "n't";
            string word_2 = word.substr(0, word.size() - 3);

            //get the n't's word vector
            MatrixXd temp_vector = MatrixXd::Zero(getVectorLength(), 1);
            getWrodVect(word_1, temp_vector);
            word_vector += temp_vector;
            
            //get the other part's word vector
            temp_vector = MatrixXd::Zero(getVectorLength(), 1);
            getWrodVect(word_2, temp_vector);
            word_vector += temp_vector;

            //get average
            word_vector /= 2;
        }
        else if (word.size() >= 3 && word.rfind("'ve") == word.size() - 3)
        {
            /*
             * if the word ends with 've, separate the 've and the other part,
             * Then add the 've vector and the word vector of the other part, and get the average
             */
            
            //get the 've and the other part of the word 
            string word_1 = "'ve";
            string word_2 = word.substr(0, word.size() - 3);

            //get the 've's word vector
            MatrixXd temp_vector = MatrixXd::Zero(getVectorLength(), 1);
            getWrodVect(word_1, temp_vector);
            word_vector += temp_vector;
            
            //get the other part's word vector
            temp_vector = MatrixXd::Zero(getVectorLength(), 1);
            getWrodVect(word_2, temp_vector);
            word_vector += temp_vector;

            //get average
            word_vector /= 2;
        }
        else if (word.size() >= 3 && word.rfind("'re") == word.size() - 3)
        {
            /*
             * if the word ends with 're, separate the 're and the other part,
             * Then add the 're vector and the word vector of the other part, and get the average
             */
            
            //get the 're and the other part of the word 
            string word_1 = "'re";
            string word_2 = word.substr(0, word.size() - 3);

            //get the 're's word vector
            MatrixXd temp_vector = MatrixXd::Zero(getVectorLength(), 1);
            getWrodVect(word_1, temp_vector);
            word_vector += temp_vector;
            
            //get the other part's word vector
            temp_vector = MatrixXd::Zero(getVectorLength(), 1);
            getWrodVect(word_2, temp_vector);
            word_vector += temp_vector;

            //get average
            word_vector /= 2;
        }
        else if (word.size() >= 2 && word.rfind("'d") == word.size() - 2)
        {
            /*
             * if the word ends with 'd, separate the 'd and the other part,
             * Then add the 'd vector and the word vector of the other part, and get the average
             */
            
            //get the 'd and the other part of the word 
            string word_1 = "'d";
            string word_2 = word.substr(0, word.size() - 2);

            //get the 'd's word vector
            MatrixXd temp_vector = MatrixXd::Zero(getVectorLength(), 1);
            getWrodVect(word_1, temp_vector);
            word_vector += temp_vector;
            
            //get the other part's word vector
            temp_vector = MatrixXd::Zero(getVectorLength(), 1);
            getWrodVect(word_2, temp_vector);
            word_vector += temp_vector;

            //get average
            word_vector /= 2;
        }
        else
        {
            /*
             * note : if not, find a word which has the smallest edit distence between the current word in the dictionary
             */
            
            string min_word = getMinEditDistenceWord(word);
            cout << word << "----" << min_word << endl;
            getWrodVect(min_word, word_vector);
        }
    }
    else
    {
        long long word_offset = word_index_dict[word];
    
        fseek(w2v_p, word_offset, 0);    //seek the pointer to the word offset relative to the head of file
    
        //skip the word at the head of the line
        while (fgetc(w2v_p) != ' ')
        {
            ;
        }
        
        //read word vector from current file pointer
        for (int row_num = 0; row_num < word_vector.rows(); ++row_num)
        {
            fscanf(w2v_p, "%lf", &word_vector(row_num, 0));
        }
    }

    fclose(w2v_p);

    return;
}

//the getter function of vector_length
int Word2VecTool::getVectorLength()
{
    return vector_length;
}
