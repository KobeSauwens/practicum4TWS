#ifndef ODE_solvers_tws
#define ODE_solvers_tws

#include <cassert>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdlib>
#include <chrono>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include "newton.hpp"

namespace ublas = boost::numeric::ublas;

enum Method
{
    EULERF,
    HEUN,
    EULERB
};

template<typename value_type, typename Op>
void euler_b_solve(ublas::matrix<value_type> & solve_values, Op const& op, value_type& rate, int nbOfGridPoints) 
{
    for(int k = 1; k < nbOfGridPoints; ++k)
    {
        #ifdef DEBUG
        std::cout << k << std::endl;
        #endif
        solve_values(k,0) = (k * rate);
        auto kr = (ublas::row(solve_values,k));
        auto kr_old = (ublas::row(solve_values,k-1));
        auto kr1 = (ublas::subrange(kr,1,solve_values.size2()));
        auto kr_old1 = (ublas::subrange(kr_old,1,solve_values.size2()));
        //solve_values(k,1) = ((k-1) * rate);
        auto initial_guess = (kr_old1 + (rate) * (op(kr_old1)));
        auto kr1_new = newton_raphson<value_type, Op>(op, kr_old1, initial_guess, true, rate);
        kr1.assign(kr1_new);
    }
}


template<typename value_type, typename Op>
void heun_solve(ublas::matrix<value_type> & solve_values, Op const& op, value_type& rate, int nbOfGridPoints) 
{
    for(int k = 1; k < nbOfGridPoints; ++k)
    {
        solve_values(k,0) = (k * rate);
        auto kr = (ublas::row(solve_values,k));
        auto kr_old = (ublas::row(solve_values,k-1));
        auto kr1 = (ublas::subrange(kr,1,solve_values.size2()));
        auto kr_old1 = (ublas::subrange(kr_old,1,solve_values.size2()));
        //solve_values(k,1) = ((k-1) * rate);
        kr1.assign(kr_old1 + rate *(double(0.5) * op(kr_old1) + (double(0.5) * op(kr_old1 + rate * op(kr_old1)))));
    }
}


template<typename value_type, typename Op>
void euler_f_solve(ublas::matrix<value_type> & solve_values, Op const& op, value_type& rate, int nbOfGridPoints) 
{

    for(int k = 1; k < nbOfGridPoints; ++k)
    {
        solve_values(k,0) = (k * rate);
        auto kr = (ublas::row(solve_values,k));
        auto kr_old = (ublas::row(solve_values,k-1));
        auto kr1 = (ublas::subrange(kr,1,solve_values.size2()));
        auto kr_old1 = (ublas::subrange(kr_old,1,solve_values.size2()));
        kr1.assign( kr_old1 + (rate) * (op(kr_old1)));
    }
}

template<typename value_type, typename Op>
void solve_ODE(ublas::matrix<value_type> & values, value_type const & horizon, Op const& op, Method method)
{
    int nbOfGridPoints = values.size1();
    value_type rate = horizon / nbOfGridPoints;

    switch(method)
    {
        case EULERF: euler_f_solve(values, op, rate, nbOfGridPoints); break;
        case HEUN: heun_solve(values, op, rate, nbOfGridPoints); break;
        case EULERB: euler_b_solve(values,op, rate, nbOfGridPoints); break;
    }

}

#endif
