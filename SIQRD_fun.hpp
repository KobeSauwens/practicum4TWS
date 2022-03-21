#ifndef SIQRD_fun_TWS_header
#define SIQRD_fun_TWS_header
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

namespace ublas = boost::numeric::ublas;

template<typename value_type>
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

ublas::vector<value_type> operator()(ublas::vector<value_type> const& v) const 
{
    assert(v.size() == 5);
    ublas::vector<value_type> f_v(5);

    value_type S = v(0);
    value_type I = v(1);
    value_type Q = v(2);
    value_type R = v(3);
    //value_type D = v(4);
    
    f_v(0) = (- beta_ * (I / (S + I + R)) * S) + (mu_ * R);
    f_v(1) = ((beta_ * (S /(S + I + R))) - gamma_ - delta_ - alpha_) * I;
    f_v(2) = (delta_ * I) - ((gamma_ + alpha_) * Q);
    f_v(3) = gamma_ * (I + Q) - (mu_ * R);
    f_v(4) = alpha_ * (I + Q);

    return f_v;
}

ublas::matrix<value_type> calc_jacobian(ublas::vector<value_type> const& v) const 
{
    assert(v.size() == 5);
    ublas::matrix<value_type> jacobian(5,5);
    jacobian.clear();

    value_type S = v(0);
    value_type I = v(1);
    //value_type Q = v(2);
    value_type R = v(3);
    //value_type D = v(4);

    jacobian(0,0) = ((I * S * beta_) / pow((I + R + S),2))-((I * beta_)/(I + R + S));
    jacobian(1,0) = I * ( (-(S * beta_)/pow((I+R+S),2))+ (beta_ / (I + R + S)) );

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

value_type& beta(){return beta_;};
value_type& mu(){return mu_;};
value_type& gamma(){return gamma_;};
value_type& alpha(){return alpha_;};
value_type& delta(){return delta_;};

private:
    value_type beta_;
    value_type mu_;
    value_type gamma_;
    value_type alpha_;
    value_type delta_;
};
#endif 