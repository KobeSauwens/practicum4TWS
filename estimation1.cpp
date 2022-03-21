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

    ublas::vector<value_type> SIQRD_params2(5);
    //beta, mu, gamma, delta, alpha
    SIQRD_params2 <<= 0.5, 0.08, 0.04, 0.09, 0.004;

    int rate = 8;
    Method method = HEUN;
    value_type eta_init = 1.;
    value_type tol = 1E-10;
    ublas::matrix<value_type> initial_B(5,5,value_type(0.));
    for(int i = 0; i < 5;++i) {initial_B(i,i) = value_type(1.);}
    value_type c_1 = 1E-4;
    value_type eps = 1E-10;

    SIQRD_fun <value_type>siqrd_fun1 = SIQRD_fun<value_type>(SIQRD_params1);
    std::string input_file = "observations1.in";
    LSE<value_type, SIQRD_fun<value_type>> LSE1(input_file, rate, eps);
    SIQRD_params1 = BFGS(SIQRD_params1, initial_B, tol, input_file, siqrd_fun1, method, eta_init, c_1, eps);
    std::cout << SIQRD_params1  << std::endl;

    ublas::matrix<value_type> initial_B2(5,5,value_type(0.));
    for(int i = 0; i < 5;++i) {initial_B2(i,i) = value_type(1.);}
    SIQRD_fun <value_type>siqrd_fun2 = SIQRD_fun<value_type>(SIQRD_params2);
    input_file = "observations2.in";
    LSE<value_type, SIQRD_fun<value_type>> LSE2(input_file, rate, eps);
    SIQRD_params2 = BFGS(SIQRD_params2, initial_B2, tol, input_file, siqrd_fun2, method, eta_init, c_1, eps);
    std::cout << SIQRD_params2  << std::endl;

    //Calculate the prediction according to the found parameters
    
    /*
    ublas::matrix<value_type> const & SIQRD_obs = LSE2.observ();
    ublas::matrix<value_type> SIQRD_values(SIQRD_obs.size1(),6,value_type(0.));
    ublas::vector<value_type> params(6);
    ublas::subrange(params,0,5) = ublas::subrange(SIQRD_params2,0,5);
    params(5) = SIQRD_obs.size1();
    SIQRD_values(0,1) = SIQRD_obs(0,1);
    SIQRD_values(0,2) = SIQRD_obs(0,2);
    siqrd_fun2.beta() = params(0);
    siqrd_fun2.mu() = params(1);
    siqrd_fun2.gamma() = params(2);
    siqrd_fun2.delta() = params(3);
    siqrd_fun2.alpha() = params(4);
    solve_ODE(SIQRD_values, params(5), siqrd_fun2, method);
    print_output(SIQRD_values, std::string("estimation2.out"));
    */
    
    return 0;
}