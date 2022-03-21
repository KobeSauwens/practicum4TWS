#ifndef LSE_TWS_header
#define LSE_TWS_header

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
#include "ODE_solvers.hpp"
#include <iomanip>
#include <limits>


namespace ublas = boost::numeric::ublas;

template<typename value_type, typename Op>
class LSE
{
    public:
    LSE(std::string file_name, int rate, value_type eps)
    :eps_( eps),
    file_name_(file_name),
    observ_( ublas::matrix<value_type>(0,0) ),
    M_( 0 ),
    N_( 0 ),
    rate_( rate )
    {
        std::string line;
        std::ifstream file(file_name_);

        std::getline(file,line);
        std::string s;
        std::istringstream split_line(line);
        split_line >> s;
        N_= stoi(s);
        split_line >> s;
        M_= stoi(s)+1;

        ublas::matrix<value_type> mat(N_,M_);
        observ_ = mat;

        for(int i = 0; i < N_; ++i) 
        {
            std::getline(file,line);
            std::istringstream split_line(line);
            for(int j = 0; j < M_; ++j)
            {
                split_line >> s;
                observ_(i,j) = stod(s);
            }
        }

    }

    LSE& operator=(LSE other)
    {
        this->observ_ = other.observ_;
        this.M_ = other.M_;
        this.N_ = other.N_;
        this->file_name_ = other.file_name_;
    }

    value_type calc_LSE(ublas::vector<value_type> params, Method method, Op op)
    {
        assert(params.size() == 5);
        ublas::matrix<value_type> SIQRD_values(rate_*N_,6);
        ublas::vector<value_type> parameters(6);
        ublas::subrange(parameters,0,5) = params;
        parameters(5) = double(N_);
        SIQRD_values(0,1) = observ_(0,1);
        SIQRD_values(0,2) = observ_(0,2);

        op.beta() = params(0);
        op.mu() = params(1);
        op.gamma() = params(2);
        op.delta() = params(3);
        op.alpha() = params(4);

        solve_ODE(SIQRD_values, value_type(N_), op, method); 

        value_type error = 0.;
        for (int i = 0; i < N_ ; ++i)
        {
            auto obs_values_i = ublas::subrange(ublas::row(observ_,i),1,6);
            auto sim_values_i = ublas::subrange(ublas::row(SIQRD_values,(rate_*i)),1,6);
            value_type pop = sum(obs_values_i);

            error += (value_type(1.)/(pow(value_type(N_),2)*pow(pop,2)))*pow(ublas::norm_2(obs_values_i - sim_values_i),2);
        }
        
        return error;
    }

    ublas::vector<value_type> calc_LSE_grad(ublas::vector<value_type> params, Method method, Op op)
    {
        ublas::vector<value_type> LSE_grad(5);
        ublas::vector<value_type> reg_params(params);


        value_type error_regular = calc_LSE(params, method, op);
        params(0) += eps_;
        value_type error_beta  = calc_LSE(params, method, op);
        params = reg_params;
        params(1) += eps_;
        value_type error_mu    = calc_LSE(params, method, op);
        params = reg_params;
        params(2) += eps_;
        value_type error_gamma = calc_LSE(params, method, op);
        params = reg_params;
        params(3) += eps_;
        value_type error_delta = calc_LSE(params, method, op);
        params = reg_params;
        params(4) += eps_;
        value_type error_alpha = calc_LSE(params, method, op);

        LSE_grad(0) = (error_beta - error_regular) / eps_;
        LSE_grad(1) = (error_mu - error_regular) / eps_;
        LSE_grad(2) = (error_gamma - error_regular) / eps_;
        LSE_grad(3) = (error_delta - error_regular) / eps_;
        LSE_grad(4) = (error_alpha - error_regular) / eps_;

        #ifdef DEBUG
        std::cout << "LSE_grad: " << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << LSE_grad << std::endl;
        #endif

        return LSE_grad;
    }

    void print_observations()
    {
        std::cout << observ_ << std::endl;
    }

    ublas::matrix<value_type> const & observ() {return observ_;}

    private:
    value_type eps_;
    std::string file_name_;
    ublas::matrix<value_type> observ_;
    int M_;
    int N_;
    int rate_;
};

#endif