#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <iostream>
#include <Python.h>
#include <string>
#include <sstream>
#include <cpprest/ws_client.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <functional>
#include <chrono>

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

using namespace boost::placeholders;
using namespace boost::property_tree;

using namespace web;
using namespace web::websockets::client;

// Converts a 2D C++ Vector into a 2D Python List
PyObject* list2D(const std::vector<std::vector<double>>& cppData) {
    PyObject* pyList = PyList_New(cppData.size());

    // Outer loop extracting each column vector
    for (size_t i = 0; i < cppData.size(); ++i) {
        PyObject* innerList = PyList_New(cppData[i].size());

        // Inner loop setting vector data into a PyList
        for (size_t j = 0; j < cppData[i].size(); ++j) {
            PyObject* value = PyFloat_FromDouble(cppData[i][j]);
            PyList_SET_ITEM(innerList, j, value);
        }

        // Setting built PyList into outer PyList
        PyList_SET_ITEM(pyList, i, innerList);
    }

    return pyList;
}

// Converts a 1D C++ Vector into a 1D Python List
PyObject* list1D(const std::vector<double>& cppData) {
    PyObject* pyList = PyList_New(cppData.size());

    // Extracting vector items and placing them into a PyList
    for (size_t i = 0; i < cppData.size(); ++i) {
        PyObject* value = PyFloat_FromDouble(cppData[i]);
        PyList_SET_ITEM(pyList, i, value);
    }

    return pyList;
}

// Parses the live price data into JSON data from the WebSocket
void Cyclone(ptree dataset, std::vector<double> & price_data)
{
    // Setting end of PTree
    ptree::const_iterator end = dataset.end();

    // Boolean checks to see if ticker has loaded
    bool ticker = false;
    for(ptree::const_iterator it = dataset.begin(); it != end; ++it){
        if(ticker == true){
            // Extracting "price" from JSON object and placing it into a vector
            if(it->first == "price"){
                price_data.push_back(atof(it->second.get_value<std::string>().c_str()));
            }
        }
        if(it->second.get_value<std::string>() == "ticker"){
            ticker = true;
        }
    }
}

// Converts a string message into a Boost PTree object for JSON parsing
ptree JSON(std::string message){
    std::stringstream ss(message);
    ptree data;
    read_json(ss, data);
    return data;
}

// Connects to a WebSocket Client in order to receive live Bitcoin price data from Coinbase
void Socket(std::vector<double> & price_data, int limit){
    // Declaring the socket address and message to subscribe to updates
    std::string url = "wss://ws-feed.exchange.coinbase.com";
    std::string msg = "{\"type\":\"subscribe\", \"product_ids\":[\"BTC-USD\"], \"channels\":[\"ticker\"]}";

    // Declaring and connecting to websocket client
    websocket_client client;
    client.connect(url).wait();

    // Sending message as utf8 to server
    websocket_outgoing_message out_msg;
    out_msg.set_utf8_message(msg);
    client.send(out_msg);

    while(true){
        // Receiving data and passing it into the Cyclone function for parsing
        client.receive().then([](websocket_incoming_message in_msg){
            return in_msg.extract_string();
        }).then([&](std::string message){
            Cyclone(JSON(message), std::ref(price_data));

            // Makes sure that vector does not exceed its limit
            if(price_data.size() > limit){
                price_data.erase(price_data.begin());
            }
        }).wait();
    }

    client.close().wait();
}

// Builds the dataframe that is used to be inputted into the Support Vector Machine
void DataFrame(std::vector<double> prices, std::vector<std::vector<double>> & X, std::vector<double> & Y)
{
    // Returns the mean and standard deviation of a given vector of prices
    auto stats = [](std::vector<double> windows)
    {
        double mean = 0, stdev = 0;
        for(auto & price : windows){
            mean += price;
        }
        mean /= (double) windows.size();
        for(auto & price : windows){
            stdev += pow(price - mean, 2);
        }
        stdev = sqrt(stdev/((double) windows.size() - 1));
        std::vector<double> result = {mean, stdev};
        return result;
    };

    // Computes the cumulative rate of return for a vector of returns to be used in classification
    auto ror = [](std::vector<double> windows)
    {
        double result = 1;
        for(int i = 1; i < windows.size(); ++i){
            result *= windows[i]/windows[i-1];
        }
        return result - 1;
    };

    // Price window and output window ex. 15 last prices & 5 future outputs
    int window = 15;
    int output = 5;

    std::vector<double> Window, Stats, Temp, OWindow;
    
    for(int i = window; i < prices.size() - output; ++i){
        Temp.clear();
        // Takes the input and output window of prices
        Window = {prices.begin() + (i - window), prices.begin() + i};
        OWindow = {prices.begin() + i, prices.begin() + i + output};

        // Computes the mean and standard deviation
        Stats = stats(Window);

        // Pushes the asset price, mean, lower bollinger, and upper bollinger bands into dataframe
        Temp.push_back(prices[i]);
        Temp.push_back(Stats[0]);
        Temp.push_back(prices[i] - 2.0*Stats[1]);
        Temp.push_back(prices[i] + 2.0*Stats[1]);
        X.push_back(Temp);
        // Decides whether to train as a long or short based on the cumulative rate of return of the OWindow
        if(ror(OWindow) > 0){
            Y.push_back(0.0);
        } else {
            Y.push_back(1.0);
        }
    }

    // Places the last elements of data for the testing set
    Window = {prices.end() - output, prices.end()};
    Stats = stats(Window);
    Temp.clear();
    Temp.push_back(prices[prices.size() - 1]);
    Temp.push_back(Stats[0]);
    Temp.push_back(prices[prices.size() - 1] - 2.0*Stats[1]);
    Temp.push_back(prices[prices.size() - 1] + 2.0*Stats[1]);
    X.push_back(Temp);

}

// Normalizes the dataframe using the ZScore method
void Normalize(std::vector<std::vector<double>> Inputs, std::vector<std::vector<double>> & NInputs)
{
    // Calculates the mean and standard deviation of the items in the Inputs dataframe
    auto stats = [](std::vector<double> windows)
    {
        double mean = 0, stdev = 0;
        for(auto & price : windows){
            mean += price;
        }
        mean /= (double) windows.size();
        for(auto & price : windows){
            stdev += pow(price - mean, 2);
        }
        stdev = sqrt(stdev/((double) windows.size() - 1));
        std::vector<double> result = {mean, stdev};
        return result;
    };

    // Transposes the dataset
    auto transpose = [](std::vector<std::vector<double>> z)
    {
        std::vector<std::vector<double>> L;
        std::vector<double> temp;
        for(int i = 0; i < z[0].size(); ++i){
            temp.clear();
            for(int j = 0; j < z.size(); ++j){
                temp.push_back(z[j][i]);
            }
            L.push_back(temp);
        }
        return L;
    };

    // Loops through each input set and normalizes each column by performing the ZScore method
    Inputs = transpose(Inputs);
    for(int i = 0; i < Inputs.size(); ++i){
        std::vector<double> Stats = stats(Inputs[i]);
        for(int j = 0; j < Inputs[i].size(); ++j){
            // ZScore Method
            Inputs[i][j] = (Inputs[i][j] - Stats[0])/Stats[1];
        }
    }
    // Places final normalized set into NInputs
    NInputs = transpose(Inputs);
}

int main()
{
    // Initializes <Python.h>
    Py_Initialize();

    // Declares all of the major variables 
    std::vector<double> pred_results;

    std::vector<std::vector<double>> Inputs, NInputs, Train, Test;
    std::vector<double> prices, outputs;

    // The prices vector will not exceed 300 prices and the start limit will allow the program to start after 
    // the prices vector has at least 60 elements in it
    int limit = 300;
    int start_limit = 60;

    // Starting the thread for the price WebSocket and it is unaffected by the calculations
    // thus allowing it to keep on going even with the sleep timer in the main program
    std::thread datafeed(Socket, std::ref(prices), limit);    

    // Loading the Support Vector Machine from Scikit-Learn
    PyObject * svm = PyImport_Import(PyUnicode_FromString("sklearn.svm"));
    PyObject * SVC = PyObject_GetAttrString(svm, "SVC");

    PyObject * EARG = PyTuple_New(0); // No arguments

    // Setting the kernel and probability
    PyObject * init_params = PyDict_New();
    PyDict_SetItemString(init_params, "kernel", PyUnicode_FromString("rbf"));
    PyDict_SetItemString(init_params, "probability", Py_True);

    // Calling the model variable
    PyObject * model = PyObject_Call(SVC, EARG, init_params);

    // Extracting functions needed to be used from the model
    PyObject * fit = PyObject_GetAttrString(model, "fit");
    PyObject * predict = PyObject_GetAttrString(model, "predict");
    PyObject * predict_prob = PyObject_GetAttrString(model, "predict_proba");

    PyObject * fit_args = PyTuple_New(2);
    PyObject * pred_args = PyTuple_New(1);
    
    while(true){
        // Waiting until start price length limit has reached the desired level
        if(prices.size() >= start_limit){
            // Clearing the input frames after each iteration to avoid having data discrepencies
            Inputs.clear();
            outputs.clear();
            NInputs.clear();
            DataFrame(prices, std::ref(Inputs), std::ref(outputs));
            Normalize(Inputs, std::ref(NInputs));

            // Splitting dataset into training and testing
            Train = {NInputs.begin(), NInputs.end() - 1};
            Test = {NInputs.end() - 1, NInputs.end()};

            // Fitting the model with the inputs and outputs
            PyTuple_SetItem(fit_args, 0, list2D(Train));    
            PyTuple_SetItem(fit_args, 1, list1D(outputs));
            PyObject_CallObject(fit, fit_args);

            // Making predictions and generating the probability that the prediction is correct
            PyTuple_SetItem(pred_args, 0, list2D(Test));
            PyObject * pred_result = PyObject_CallObject(predict, pred_args);
            PyObject * prob_result = PyObject_CallObject(predict_prob, pred_args);

            // Extracting the Python output from the Support Vector Machine back into C++
            Py_ssize_t size1 = PySequence_Size(pred_result);
            Py_ssize_t size2 = PySequence_Size(prob_result);

            std::vector<double> OUTPUT;
            for(Py_ssize_t i = 0; i < size1; ++i){
                OUTPUT.push_back(PyFloat_AsDouble(PySequence_GetItem(pred_result, i)));  
            }
            for(Py_ssize_t i = 0; i < size2; ++i){
                PyObject * temporary = PySequence_GetItem(prob_result, i);
                for(Py_ssize_t j = 0; j < PySequence_Size(temporary); ++j){
                    OUTPUT.push_back(PyFloat_AsDouble(PySequence_GetItem(temporary, j)));
                }
            }

            // Printing out what the classification and probability the SVM chose to output based on its training
            if(OUTPUT[0] == 0){
                std::cout << "The chance of a Long working is " << OUTPUT[1] << std::endl;
            } else {
                std::cout << "The chance of a Short working is " << OUTPUT[2] << std::endl;
            }
            
            // Main thread sleeps for 3 seconds, WebSocket thread still works while this thread is sleeping
            std::this_thread::sleep_for(std::chrono::seconds(3));
        } else {
            std::cout << "Prices left to load: " << start_limit - prices.size() << std::endl;
        }
    }

    // Finalizes <Python.h>
    Py_Finalize();

    return 0;
}