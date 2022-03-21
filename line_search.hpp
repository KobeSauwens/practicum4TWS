#ifndef LINE_SEARCH_TWS_HEADER
#define LINE_SEARCH_TWS_HEADER

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
#include "LSE.hpp"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/assignment.hpp>


namespace ublas = boost::numeric::ublas;

template<typename value_type, typename Op>
value_type line_search(value_type eta_init, value_type c_1, ublas::vector<value_type> d_k, ublas::vector<value_type> p_k, Method method, Op op, value_type eps, LSE<value_type,Op> LSE_siqrd)
{
    value_type eta = eta_init;
    value_type LSE_p_k = LSE_siqrd.calc_LSE(p_k, method, op);
    ublas::vector<value_type> grad_LSE_p_k = LSE_siqrd.calc_LSE_grad(p_k, method, op);
    value_type c_1_d_k_grad_LSE_p_k = c_1 * ublas::inner_prod(d_k,grad_LSE_p_k);

    while(LSE_siqrd.calc_LSE(p_k+(eta*d_k), method, op) > (LSE_p_k + eta * c_1_d_k_grad_LSE_p_k) )
    {
        eta = eta / value_type(2.);
    }

    return eta;
}

#endif