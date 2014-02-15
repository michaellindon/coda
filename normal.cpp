#include <cstdlib>
#include <iostream>
#include <vector>
#include <new>
#include <armadillo>

using namespace std;
using namespace arma;

extern "C" void normal(){
	std::cout << "Printing to R test" << endl;
	  mat A = randu<mat>(4,5);
	    mat B = randu<mat>(4,5);
	      cout << A*B.t() << endl;
}
