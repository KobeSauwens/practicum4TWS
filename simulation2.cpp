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
#include "euler_f.hpp"
#include "euler_b.hpp"
#include "newton.hpp"
#include "heun.hpp"


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

template<typename param_type, typename elem_type>
struct SIQRD_fun 
{
public:
SIQRD_fun( ublas::vector<value_type> parameters )
:beta_( parameters(0) )
,mu_( parameters(1) )
,gamma_( parameters(2) )
,alpha_( parameters(3) )
,delta_( parameters(4) ) 
{}

ublas::vector<elem_type> operator()(ublas::vector<elem_type> const& v) const 
{
    assert(v.size() == 5);
    ublas::vector<elem_type> f_v(5);

    elem_type S = v(0);
    elem_type I = v(1);
    elem_type Q = v(2);
    elem_type R = v(3);
    //elem_type D = v(4);
    
    f_v(0) = (- beta_ * (I / (S + I + R)) * S) + (mu_ * R);
    f_v(1) = ((beta_ * (S /(S + I + R))) - gamma_ - delta_ - alpha_) * I;
    f_v(2) = (delta_ * I) - ((gamma_ + alpha_) * Q);
    f_v(3) = gamma_ * (I + Q) - (mu_ * R);
    f_v(4) = alpha_ * (I + Q);

    return f_v;
}

ublas::matrix<elem_type> calc_jacobian(ublas::vector<elem_type> const& v) const 
{
    assert(v.size() == 5);
    ublas::matrix<elem_type> jacobian(5,5);
    jacobian.clear();

    elem_type S = v(0);
    elem_type I = v(1);
    //elem_type Q = v(2);
    elem_type R = v(3);
    //elem_type D = v(4);

    jacobian(0,0) = ((I * S * beta_) / pow((I + R + S),2))- ((I * beta_)/(I + R + S));
    jacobian(1,0) = I * ( (-(S * beta_)/pow((I+R+S),2)))+ (beta_ / (I + R + S));

    jacobian(0,1) = ((I * S * beta_) / pow((I + R + S),2)) - ((S * beta_)/(I + R + S));
    jacobian(1,1) = - ((I * S * beta_) / pow((I + R + S),2)) + ((S * beta_)/(I + R + S)) - alpha_ - delta_ - gamma_;
    jacobian(2,1) = delta_;
    jacobian(3,1) = gamma_;
    jacobian(4,1) = alpha_;

    jacobian(2,2) = - alpha_ - gamma_;
    jacobian(3,2) = gamma_;
    jacobian(4,2) = alpha_;
    
    jacobian(0,3) = ((I * S * beta_) / pow((I + R + S),2)) + mu_;
    jacobian(1,3) = - ((I * S * beta_) / pow((I + R + S),2));
    jacobian(3,3) = - mu_;
    
    return jacobian;
}

private:
    elem_type beta_;
    elem_type mu_;
    elem_type gamma_;
    elem_type alpha_;
    elem_type delta_;
};

int main(int argc, char **argv) 
{
    double     N = std::stoi(argv[1]);
    double     T = std::stod(argv[2]);
    ublas::vector<value_type> SIQRD_params(8);
    ublas::matrix<value_type> SIQRD_values(N,6,double(0.));

    //beta
    SIQRD_params(0) = 0.5;
    //mu
    SIQRD_params(1) = 0.;
    //gamma
    SIQRD_params(2) = 0.2;
    //alpha
    SIQRD_params(3) = 0.005;
    //delta
    SIQRD_params(4) = 0.;
    //S_0
    SIQRD_params(5) = 100;
    //I_0
    SIQRD_params(6) = 5;
    //T
    SIQRD_params(7) = T;

    SIQRD_values(0,1) = 100;
    SIQRD_values(0,2) = 5;

    enum Method
    {
        EULERF,
        HEUN,
        EULERB
    };

    Method method = EULERB;
    SIQRD_fun <value_type,value_type>siqrd_fun = SIQRD_fun<value_type,value_type>(SIQRD_params);


    std::string file_name;
    switch(method)
    {
        case EULERF:    
            file_name = "fwe_no_measures.txt";//"euler_f_output.txt";
            euler_f_solve(SIQRD_values, SIQRD_params, siqrd_fun); 
            break;

        case HEUN:  
            file_name = "bwe_quarantine.txt";//"heun_output.txt";
            heun_solve(SIQRD_values, SIQRD_params, siqrd_fun);
            break;

        case EULERB: 
            file_name = "heun_lockdown.txt";//"euler_b_output.txt";
            euler_b_solve(SIQRD_values, SIQRD_params, siqrd_fun);
            break;
    }
    
    print_output(SIQRD_values, file_name);
}
