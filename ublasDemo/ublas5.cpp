#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <iostream>

namespace ublas = boost::numeric::ublas;

int main()
{
    ublas::identity_matrix<double> I(3);
    std::cout<<I<<std::endl;

    ublas::matrix<double> M(3,3), M2(3,3);
    M.assign(I);
    std::cout<<M<<std::endl;

    ublas::vector<double> v(2),v2(2);
    v(0) = 3.; v(1) = 4.;
    std::cout<<ublas::norm_2(v)<<std::endl;

    v*= 0.1;
    v2.assign(v*0.1);
    std::cout<<v<<" "<<v2<<std::endl;
    M*=0.1;
    M2.assign(M*0.1);
    std::cout<<M<<" "<<M2<<std::endl;

    M.clear();
    std::cout<<M<<std::endl;

    // Power of a number
std::cout<<std::pow(3.,2)<<std::endl; // prints 9.
// Square root of a number
std::cout<<std::sqrt(9.)<<std::endl; // prints 3.
}
