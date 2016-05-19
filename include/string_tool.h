#ifndef STRING_TOOL_H
#define STRING_TOOL_H

#include<iostream>
#include<vector>

using namespace std;

class StringTool
{

public:

    void split(const string&, string, vector<string>&);    //split a string with geiven delim
    int calEditDistence(const string&, const string&);    //calculate the edit distence between two wrod
    vector<string> tokenize(const string, string delimitation = " ,.?!()[]{}<>~#&*`\";");    //tokenize the sentence into a vector with the geiven delimitation

};
  
#endif
