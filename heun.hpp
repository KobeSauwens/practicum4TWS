#ifndef heun_tws
#define heun_tws

#include <cassert>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdlib>
#include <chrono>

namespace ublas = boost::numeric::ublas;

template<typename value_type, typename Op>
void heun_solve(ublas::matrix<value_type> & SIQRD_values, ublas::vector<value_type> const& parameters, Op const& op) 
{
    assert(parameters.size() == 8);
    assert(SIQRD_values.size2() == 6);
    value_type horizon = parameters(7);
    value_type nbOfGridPoints = SIQRD_values.size1();
    value_type rate = horizon / nbOfGridPoints;

    for(int k = 1; k < int(ceil(nbOfGridPoints)); ++k)
    {
        SIQRD_values(k,0) = k * rate;
        auto kr = ublas::row(SIQRD_values,k);
        auto kr_old = ublas::row(SIQRD_values,k-1);
        auto kr1 = ublas::subrange(kr,1,6);
        auto kr_old1 = ublas::subrange(kr_old,1,6);
        SIQRD_values(k,1) = (k-1) * rate;
        kr1 = kr_old1 + rate *(double(0.5) * op(kr_old1) + (double(0.5) * op(kr_old1 + rate * op(kr_old1))));
    }
}
#endif
