#ifndef BFGS_TWS_header
#define BFGS_TWS_header
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
#include "line_search.hpp"
#include "SIQRD_fun.hpp"


namespace ublas = boost::numeric::ublas;


template <typename value_type, typename Op>
ublas::vector<value_type> BFGS(ublas::vector<value_type> p_k, ublas::matrix<value_type> B_k, value_type tol, std::string input_file, Op op, Method method, value_type eta_init, value_type c_1, value_type eps)
{
    int m = p_k.size();
    SIQRD_fun <value_type>siqrd_fun = SIQRD_fun<value_type>(p_k);
    LSE <value_type, SIQRD_fun<value_type>> LSE_siqrd = LSE<value_type, SIQRD_fun<value_type>>(input_file, 8,eps);
    ublas::vector<value_type> p_k_new(m), s(m), d_k(m), LSE_grad(m), LSE_grad_new(m), y(m);
    ublas::matrix<value_type> B_k_c(m,m);
    ublas::permutation_matrix<int> pm1(B_k.size1());
    ublas::permutation_matrix<int> pm2(B_k.size1());
    value_type eta;
    int iter = 0;
    while(true)
    {
        ++iter;
        #ifdef DEBUG
        std::cout << "LSE: " << LSE_siqrd.calc_LSE(p_k,method,siqrd_fun) << std::endl;
        #endif
        LSE_grad = LSE_siqrd.calc_LSE_grad(p_k, method, siqrd_fun);
        d_k.assign(LSE_grad);
        B_k_c.assign(B_k);
        ublas::lu_factorize(B_k_c,pm1);
        ublas::lu_substitute(B_k_c, pm1, d_k);
        d_k.assign(-d_k);
        eta = line_search(eta_init, c_1, d_k, p_k, method, op, eps, LSE_siqrd);
        #ifdef DEBUG
        std::cout << "eta: " << eta << std::endl;
        #endif
        p_k_new.assign(p_k + (eta * d_k));
        s.assign(p_k_new - p_k);
        if((ublas::norm_2(s) / ublas::norm_2(p_k)) < tol)
        {
            ublas::vector<value_type> result(6,0.);
            ublas::subrange(result,0,5) = p_k_new;
            result(5) = iter;
            return result;
        }
        LSE_grad_new.assign(LSE_siqrd.calc_LSE_grad(p_k_new, method, siqrd_fun));
        y.assign(LSE_grad_new - LSE_grad);

        ublas::vector<value_type> matvec_B_k_s = (ublas::prod(B_k,s));
        ublas::matrix<value_type> outer_matvec_B_k_s__s = (ublas::outer_prod(matvec_B_k_s,s));
        ublas::matrix<value_type> B_k_t = (ublas::trans(B_k));

        B_k.assign(
        B_k + ( ublas::outer_prod(y,y) / ublas::inner_prod(s,y) ) 
        - 
        (ublas::prod( outer_matvec_B_k_s__s, B_k_t ) 
        / 
        ( ublas::inner_prod(s,matvec_B_k_s) ) 
        )
        );
        #ifdef DEBUG
        std::cout << "B_k" << B_k << std::endl;
        std::cout << "p_k_new: " << p_k_new << std::endl;
        #endif 
        p_k.assign(p_k_new);
        pm1.assign(pm2);
    }
}


#endif