#include <cstdlib>
#include <iostream>
#include <vector>
#include <new>
#include <armadillo>

using namespace std;
using namespace arma;

extern "C" void normal(double *ryo, double *rxo, int *rno, int *rp){

//	mat xa(*rp,*rp);
	colvec yo(*rno);
//	colvec ya(*rp);


 std::copy(ryo, ryo + *rno, &yo(0));
//	cout << yo << endl;
	cout << yo << endl;
	cout << yo(2) << endl;
}
