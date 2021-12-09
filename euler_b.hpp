#ifndef euler_b_tws
#define euler_b_tws

#include <cassert>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdlib>
#include <chrono>
#include "newton.hpp"

namespace ublas = boost::numeric::ublas;

template<typename value_type, typename Op>
void euler_b_solve(ublas::matrix<value_type> & SIQRD_values, ublas::vector<value_type> const& parameters, Op const& op) 
{
    assert(parameters.size() == 8);
    assert(SIQRD_values.size2() == 6);
    value_type horizon = parameters(7);
    value_type nbOfGridPoints = SIQRD_values.size1();
    value_type rate = horizon / nbOfGridPoints;

    for(int k = 1; k < int(ceil(nbOfGridPoints)); ++k)
    {
        std::cout << k << std::endl;
        SIQRD_values(k,0) = k * rate;
        auto kr = ublas::row(SIQRD_values,k);
        auto kr_old = ublas::row(SIQRD_values,k-1);
        auto kr1 = ublas::subrange(kr,1,6);
        auto kr_old1 = ublas::subrange(kr_old,1,6);
        SIQRD_values(k,1) = (k-1) * rate;
        auto initial_guess = kr_old1 + (rate) * (op(kr_old1));
        auto final_guess = newton_raphson<value_type, Op>(op, kr_old1, initial_guess, true, rate);
        kr1 = final_guess;
    }
}

#endif
