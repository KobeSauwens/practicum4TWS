#include <boost/numeric/ublas/vector.hpp>
namespace ublas = boost::numeric::ublas;

int main(){
ublas::vector<double> x(10,0.1), y(10), z(10);
for (unsigned int i=0; i<1000000; i++){
y.assign(x+z);
z.assign(y+x);
}
}
