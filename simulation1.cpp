#include <cassert>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdlib>
#include <chrono>
#include <string>
#include <cstring>
#include <fstream> 
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include "newton.hpp"
#include "ODE_solvers.hpp"
#include "SIQRD_fun.hpp"


namespace ublas = boost::numeric::ublas;
typedef double value_type;

template<typename element_type>
void print_output(ublas::matrix<element_type> const & SIQRD_values, std::string file_name)
{
    std::ofstream outputFile(file_name); 
    for(unsigned int i = 0; i < SIQRD_values.size1(); ++i)
    { 
        for(unsigned int j = 0; j < SIQRD_values.size2(); ++j)
        {
            outputFile << SIQRD_values(i,j) << " ";
        }
        outputFile << std::endl;
    }
}


int main(int argc, char **argv) 
{
    value_type     N = std::stoi(argv[1]);
    value_type     T = std::stod(argv[2]);
    ublas::vector<value_type> SIQRD_params(6);
    ublas::matrix<value_type> SIQRD_values(N,6,double(0.));

    //beta
    SIQRD_params(0) = 0.5;
    //mu
    SIQRD_params(1) = 0;
    //gamma
    SIQRD_params(2) = 0.2;
    //delta
    SIQRD_params(3) = 0.9;
    //alpha
    SIQRD_params(4) = 0.005;
    //T
    SIQRD_params(5) = T;

    SIQRD_values(0,1) = 100;
    SIQRD_values(0,2) = 5;

    Method method = HEUN;
    SIQRD_fun <value_type>siqrd_fun = SIQRD_fun<value_type>(SIQRD_params);


    std::string file_name;
    switch(method)
    {
        case EULERF:    
            file_name = "fwe_no_measures.txt";//"euler_f_output.txt";
            break;

        case HEUN:  
            file_name = "heun_lockdown.txt";//"heun_output.txt";
            break;

        case EULERB: 
            file_name = "bwe_quarantine.txt";//"euler_b_output.txt";
            break;
    } 
    solve_ODE(SIQRD_values, T, siqrd_fun, method);
    print_output(SIQRD_values, file_name);
}
