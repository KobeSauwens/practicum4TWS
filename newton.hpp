#ifndef newton_tws
#define newton_tws

#include <cassert>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdlib>
#include <chrono>
#include <boost/numeric/ublas/lu.hpp>


namespace ublas = boost::numeric::ublas;

template<typename Op, typename value_type>

value_type calculate_error(Op op, ublas::vector<value_type> const & prev_point, ublas::vector<value_type> const & estimation, value_type ratio, bool relative)
{
    value_type abs_error = ublas::norm_2(prev_point + ratio * op(estimation) - estimation);
    if (relative)
    {
        return abs_error / ublas::norm_2(estimation);    
    }else 
    {
        return abs_error;
    }
}

template<typename value_type, typename Op>
ublas::vector<value_type> newton_raphson(Op op, ublas::vector<value_type> prevPoint, ublas::vector<value_type> current_guess, bool relative, value_type ratio)
{ 
    int const max_iter = 500;
    value_type tolerance = 10E-10;
    ublas::permutation_matrix<int> pm1(prevPoint.size());
    ublas::permutation_matrix<int> pm2(prevPoint.size());
    ublas::matrix<value_type> I(prevPoint.size(),prevPoint.size(), 0.);
    for(unsigned int i = 0; i < prevPoint.size(); ++i) {I(i,i) = value_type(1.);}
    ublas::vector<value_type> b(prevPoint.size());
    ublas::matrix<value_type> jacobian(prevPoint.size(),prevPoint.size()); 
    for(int iter = 1; iter < max_iter; ++iter)
    {
        #ifdef DEBUG
        std::cout << "Newton Error: " << calculate_error<Op, value_type>(op, prevPoint, current_guess, relative, ratio) << std::endl
        << "Current guess: " << current_guess << std::endl
        << "prevPoint: " << prevPoint << std::endl;
        #endif
        if (calculate_error<Op, value_type>(op, prevPoint, current_guess, relative, ratio) < tolerance )
        {
            return current_guess;
        }
        b.assign(prevPoint + (ratio) * op(current_guess) - current_guess);
        jacobian.assign(op.calc_jacobian(current_guess));
        #ifdef DEBUG
        std::cout << jacobian << std::endl;
        #endif
        jacobian.assign(ratio * jacobian);
        jacobian.assign(jacobian - I);
        ublas::lu_factorize(jacobian,pm1);
        ublas::lu_substitute(jacobian, pm1, b);
        current_guess.assign(current_guess - b);
        #ifdef DEBUG
        std::cout << current_guess << std::endl;
        #endif
        pm1.assign(pm2);
    }
    #ifdef DEBUG
    std::cout << "Warning: Maximum iterations of (" << max_iter << ") reached" << std::endl;
    #endif
    ublas::vector<value_type> v(current_guess.size());
    return v;
}

template<typename value_type, typename Op, typename Op_grad>
ublas::vector<value_type> newton_raphson(Op op, Op_grad grad, ublas::vector<value_type> prevPoint, ublas::vector<value_type> current_guess, bool relative, value_type ratio)
{ 
    int const max_iter = 500;
    value_type tolerance = pow(10.,-10);
    ublas::permutation_matrix<int> pm1(prevPoint.size());
    ublas::permutation_matrix<int> pm2(prevPoint.size());
    ublas::identity_matrix<value_type> I(prevPoint.size());
    ublas::vector<value_type> b(prevPoint.size());
    ublas::matrix<value_type> jacobian(prevPoint.size(),prevPoint.size()); 
    for(int iter = 1; iter < max_iter; ++iter)
    {
        #ifdef DEBUG
        std::cout << calculate_error<Op, value_type>(op, prevPoint, current_guess, relative, ratio)<< std::endl;
        #endif
        if (calculate_error<Op, value_type>(op, prevPoint, current_guess, relative, ratio) < tolerance )
        {
            return current_guess;
        }
        b.assign(prevPoint + (ratio) * op(current_guess) - current_guess);
        jacobian.assign(grad(current_guess));
        jacobian.assign(ratio * jacobian);
        jacobian.assign(jacobian - I);
        ublas::lu_factorize(jacobian,pm1);
        ublas::lu_substitute(jacobian, pm1, b);
        current_guess.assign(current_guess - b);
        pm1.assign(pm2);
    }
    #ifdef DEBUG
    std::cout << "Warning: Maximum iterations of (" << max_iter << ") reached";
    #endif
    ublas::vector<value_type> v(current_guess.size());
    return v;
}


#endif
