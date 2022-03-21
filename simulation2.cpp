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


namespace ublas = boost::numeric::ublas;
typedef double value_type;

struct Weird_ODE2 
{
public:
Weird_ODE2()
{}

ublas::vector<value_type> operator()(ublas::vector<value_type> v) const 
{
    ublas::vector<value_type> & f_v(v);
    
    std::generate(f_v.begin(), f_v.end(), [n = -1,f_v] () mutable {n++; return -10 * pow(f_v(n)-(value_type(n)/100.),3); });
    return f_v; 
}

ublas::matrix<value_type> calc_jacobian(ublas::vector<value_type> v) const 
{
    std::generate(v.begin(), v.end(), [n = -1,v] () mutable {n++; return -30 * pow(v(n)-(value_type(n)/100.),2); });
    ublas::matrix<value_type> jacobian(v.size(),v.size(),0.);
    for(long unsigned int i= 0; i < v.size();++i )
    {
       jacobian(i,i) = v(i);
    }
    return jacobian;
}
};

ublas::vector<value_type> weird_ODE3(ublas::vector<value_type> v)
{
    std::generate(v.begin(), v.end(), [n = -1,v] () mutable {n++; return -10 * pow(v(n)-(value_type(n)/100.),3); }); 
    return v;
}

template<typename element_type>
void print_output(ublas::matrix<element_type> const & grid_values, std::string file_name)
{
    std::ofstream outputFile(file_name); 
    for(unsigned int i = 0; i < grid_values.size1(); i += 100)
    { 
        outputFile << grid_values(i,0) << " " << grid_values(i,1) << " " << grid_values(i,25) << " " << grid_values(i,50) << std::endl; 
    }
}

int main(int argc, char **argv)
{
    value_type N = std::stoi(argv[1]);
    value_type T = std::stod(argv[2]);
    ublas::matrix<value_type> grid_values(N, 51, value_type(0.));

    auto row0 = ublas::row(grid_values,0);
    auto row0_values = ublas::subrange(row0,1,51);
    std::iota(row0_values.begin(),row0_values.end(),1);
    row0_values = row0_values*value_type(0.01); 

    auto weird_ODE = [](ublas::vector<value_type> v) -> ublas::vector<value_type>
    {std::generate(v.begin(), v.end(), [n = -1,v] () mutable {n++; return -10 * pow(v(n)-(value_type(n)/100.),3); });return v; };

    int nbOfGridPoints = grid_values.size1();
    value_type rate = T / nbOfGridPoints;

    std::string file_name = "heun_sim2.txt";
    heun_solve(grid_values, weird_ODE, rate, nbOfGridPoints);
    print_output<value_type>(grid_values, file_name);

    std::iota(row0_values.begin(),row0_values.end(),1);
    row0_values = row0_values*value_type(0.01); 
    Weird_ODE2 weird_ODE2 = Weird_ODE2();
    file_name = "fwe_sim2.txt";
    euler_f_solve(grid_values, weird_ODE3, rate, nbOfGridPoints);
    print_output<value_type>(grid_values, file_name);

    ublas::matrix<value_type> grid_values3(N, 51, value_type(0.));
    row0 = ublas::row(grid_values3,0);
    row0_values = ublas::subrange(row0,1,51);
    std::iota(row0_values.begin(),row0_values.end(),1);
    row0_values = row0_values*value_type(0.01); 
    file_name = "bwe_sim2.txt";
    //solve_ODE(grid_values3, T, weird_ODE2, EULERB);
    print_output<value_type>(grid_values3, file_name);
}
