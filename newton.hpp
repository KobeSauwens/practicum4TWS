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
/*     logical, intent(in)                     :: relative
    real(RPN)                               :: horizon
    real(RPN),intent(in), dimension(5)      :: prevPoint
    real(RPN),intent(in), dimension(6)      :: parameters
    real(RPN),intent(inout), dimension(5)   :: currentGuess
    real(RPN), dimension(5)                 :: litteVector
    real(RPN), dimension(5,5)               :: jacobian
    integer(IPN),intent(in)                 :: nbOfGridPoints
    integer(IPN)                            :: iter, j */

    int const max_iter = 500;
    value_type tolerance = pow(10.,-10);

    for(int iter = 1; iter < max_iter; ++iter)
    {
        if (calculate_error<Op, value_type>(op, prevPoint, current_guess, relative, ratio) < tolerance )
        {
            return current_guess;
        }
        ublas::vector<value_type> b(prevPoint + (ratio) * op(current_guess) - current_guess);
        ublas::matrix<value_type> jacobian(op.calc_jacobian(current_guess));
        jacobian.assign(ratio * jacobian);
        ublas::identity_matrix<value_type> I(5);
        jacobian.assign(jacobian - I);
        ublas::permutation_matrix<int> pm(jacobian.size1());
        ublas::lu_factorize(jacobian,pm);
        ublas::lu_substitute(jacobian, pm, b);
        current_guess = current_guess - b;
    }
    std::cout << "Warning: Maximum iterations of (" << max_iter << ") reached";
    ublas::vector<value_type> v(current_guess.size());
    return v;
}


#endif
