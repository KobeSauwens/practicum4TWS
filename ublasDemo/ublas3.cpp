#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
namespace ublas = boost::numeric::ublas;
int main(int argc, char *argv[]){
	ublas::matrix<double> A(3,3);
	ublas::vector<double> v(3);
	v(0) = 1; v(1) = 2; v(2) = 4;
	ublas::matrix_row<ublas::matrix<double>>a0(A,0);
	std::cout<<a0<<std::endl;
	a0.assign(v);
	std::cout<<a0<<std::endl;
	std::cout<<A<<std::endl;
	ublas::matrix_column<ublas::matrix<double>>ac0(A,0);
	std::cout<<ac0<<std::endl;
	ac0.assign(v);
	std::cout<<A<<std::endl;
	return 0;
}
