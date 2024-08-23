# BitcoinSVM
This program utilizes a Support Vector Machine to classify whether to go Long or Short based on price data patterns. Additionally, the probability of the classification being correct is included in the output. This was all written in C++ but uses Python Objects to load Scikit-Learn and trains the model on live data.

## Installation
The installation shell script ```go.sh``` allows you to compile the program. The libraries it uses are <Python.h>, <Cpprest>, and <Boost>. You will also need to have Scikit-Learn installed with your version of Pip in order for it to be callable from the C++ program.

## Running
![alt](https://github.com/mosharieff47/BitcoinSVM/blob/main/results.png)
