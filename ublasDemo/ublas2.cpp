#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <cstdlib>

namespace ublas = boost::numeric::ublas;

void random_numbers(ublas::matrix<double>& A)
{
for (decltype(A.size1()) i = 0;  i < A.size1(); ++ i) {
    for (decltype(A.size2()) j = 0;  j< A.size2(); ++j) {
        A(i,j) = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
    }
}
}

int main(int argc, char *argv[]){
	ublas::matrix<double> A1(2,2);
	A1(0,0) = 1; A1(0,1) = 1;
	A1(1,0) = 1; A1(1,1) = -1;
	ublas::vector<double> x1(2);
	x1(0) = 2; x1(1) = 0;
	ublas::permutation_matrix<size_t> pm(A1.size1());
	ublas::lu_factorize(A1,pm);
	ublas::lu_substitute(A1, pm, x1);
	std::cout<<x1<<std::endl;


    size_t n = 20;
    ublas::vector<double> x(n), b(n), r(n);
    b(0) = 1;
    ublas::matrix<double> A(n,n), A2(n,n);
    ublas::permutation_matrix<size_t> pm1(A.size1());
    ublas::permutation_matrix<size_t> pm2(A.size1());
    for (size_t i = 0;i<100;i++){
        random_numbers(A);
        A2.assign(A);
        ublas::lu_factorize(A,pm1);
        x.assign(b);
        ublas::lu_substitute(A, pm1, x);
        r.assign(ublas::prod(A2,x));
        r-=b;
        std::cout<<ublas::norm_2(r)<<std::endl;
        pm1.assign(pm2); // Removing this line will lead to wrong results
    }
	return 0;
}
