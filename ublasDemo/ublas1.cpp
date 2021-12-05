#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <iostream>
#include <boost/numeric/ublas/assignment.hpp>

namespace ublas = boost::numeric::ublas;

int main(int argc, char *argv[]){
	ublas::matrix<double> A(3,3); // matrix
	A(0,0) = 1; A(0,1) = 1; A(0,2) = 1;
	A(1,0) = 1; A(1,1) = -1; A(1,2) = 0;
	A(2,0) = 1; A(2,1) = 0; A(2,2) = -1;
	ublas::vector<double> x(3); // vector
	x(0) = 1; x(1) = 2; x(2) = 3;
	std::cout<<A<<std::endl; // io
	std::cout<<x<<std::endl;
    ublas::matrix<double> A1(3,3); // matrix
    A1<<=1,1,1,1,-1,0,1,0,-1;
    ublas::vector<double> x1(3); // vector
    x1<<=1,2,3;
    std::cout<<A1<<std::endl;
    std::cout<<x1<<std::endl;
    std::cout<<x.size()<<std::endl;
    std::cout<<A1.size1()<<std::endl;
    std::cout<<A1.size2()<<std::endl;


    ublas::vector<double> v1(3),v2(3);
    v1<<= 1,2,3;
    v2<<= 3,2,1;
    double alpha = ublas::inner_prod(v1,v2);
    std::cout<<alpha<<std::endl;

    ublas::matrix<double> P(ublas::outer_prod(v1,v2));
    std::cout<<P<<std::endl;

    v2.assign(ublas::prod(A,v1));
    std::cout<<v2<<std::endl;
	return 0;
}
