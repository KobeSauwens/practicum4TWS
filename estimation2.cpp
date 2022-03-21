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
#include "BFGS.hpp"
#include "LSE.hpp"
#include "SIQRD_fun.hpp"
#include <boost/numeric/ublas/assignment.hpp>
#include <iomanip>
#include <ctime>
#include <thread>


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

int main() 
{
    ublas::vector<value_type> SIQRD_params1(5);
    //beta
    SIQRD_params1(0) = 0.32;
    //mu
    SIQRD_params1(1) = 0.03;
    //gamma
    SIQRD_params1(2) = 0.151;
    //delta
    SIQRD_params1(3) = 0.052;
    //alpha
    SIQRD_params1(4) = 0.004;

    int rate = 8;
    value_type eta_init = 1.;
    value_type tol = 1E-10;
    ublas::matrix<value_type> initial_B(5,5,value_type(0.));
    for(int i = 0; i < 5;++i) {initial_B(i,i) = value_type(1.);}
    value_type c_1 = 1E-4;
    value_type eps = 1E-10;
    std::string input_file = "observations1.in";
    SIQRD_fun <value_type>siqrd_fun = SIQRD_fun<value_type>(SIQRD_params1);
    LSE<value_type, SIQRD_fun<value_type>> LSE1(input_file, rate, eps);


    //Warm up
    ublas::vector<value_type> SIQRD_params2_W = BFGS(SIQRD_params1, initial_B, tol, input_file, siqrd_fun, EULERF, eta_init, c_1, eps);
    
    std::clock_t c_start_F = std::clock();
    ublas::vector<value_type> SIQRD_params2_F = BFGS(SIQRD_params1, initial_B, tol, input_file, siqrd_fun, EULERF, eta_init, c_1, eps);
    std::clock_t c_end_F = std::clock();
    std::clock_t c_start_H = std::clock();
    ublas::vector<value_type> SIQRD_params2_H = BFGS(SIQRD_params1, initial_B, tol, input_file, siqrd_fun, HEUN, eta_init, c_1, eps);
    std::clock_t c_end_H = std::clock();
    //std::clock_t c_start_B = std::clock();
    //ublas::vector<value_type> SIQRD_params2_B = BFGS(SIQRD_params1, initial_B, tol, input_file, siqrd_fun, EULERB, eta_init, c_1, eps);
    //std::clock_t c_end_B = std::clock();

    std::cout << std::setprecision(0)
    << "Forward Euler:" << std::endl 
    << " - Number of BFGS iterations: " << SIQRD_params2_F(5) << std::endl
    << std::setprecision(8) 
    << " - Execution time: " << std::fixed << value_type(c_end_F - c_start_F) / CLOCKS_PER_SEC << " seconds" << std::endl
    << " - Obtained parameters" << "(" << SIQRD_params2_F(0) << "," << SIQRD_params2_H(1) << ","  << SIQRD_params2_H(2) << ","  << SIQRD_params2_H(3) << ","  << SIQRD_params2_H(4) << ")" << std::endl 
    //<< "--------------------------------------" << std::endl
    //<< "Backward Euler:" << std::endl 
    //<< "Implementation error..." << std::endl
    //<< " - Number of BFGS iterations: " << SIQRD_params2_B(5) << std::endl
    //<< " - Execution time: " << value_type(c_end_B - c_start_B)/CLOCKS_PER_SEC << " seconds" << std::endl
    //<< " - Obtained parameters" << ublas::subrange(SIQRD_params2_B,0,5) << std::endl 
    << "--------------------------------------" << std::endl
    << "Heun's method:" << std::endl 
    << std::setprecision(0) 
    << " - Number of BFGS iterations: " << SIQRD_params2_H(5) << std::endl
    << std::setprecision(8) 
    << " - Execution time: " << value_type(c_end_H - c_start_H) / CLOCKS_PER_SEC << " seconds" << std::endl
    << " - Obtained parameters" << "(" << SIQRD_params2_H(0) << "," << SIQRD_params2_H(1) << ","  << SIQRD_params2_H(2) << ","  << SIQRD_params2_H(3) << ","  << SIQRD_params2_H(4) << ")" << std::endl; 

    return 0;
}