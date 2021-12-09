#ifndef euler_f_tws
#define euler_f_tws

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

namespace ublas = boost::numeric::ublas;

template<typename value_type, typename Op>
void euler_f_solve(ublas::matrix<value_type> & SIQRD_values, ublas::vector<value_type> const& parameters, Op const& op) 
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
        kr1.assign( kr_old1 + (rate) * (op(kr_old1)));
    }
}
#endif