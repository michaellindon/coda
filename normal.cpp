#include <cstdlib>
#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

extern "C" void normal(double *ryo, double *rxo, int *rno, int *rp, int *niter){

	mat xa(*rp,*rp);
	mat xaxa(*rp,*rp);
	mat xo(*rno,*rp);
	colvec yo(*rno);
	vec gamma(*rp,fill::ones);

 std::copy(ryo, ryo + *rno, &yo(0));
 std::copy(rxo, rxo+(*rno * *rp), &xo(0,0));

 xaxa=xo.t()*xo;
 xaxa.diag()=vec(xaxa.n_rows,fill::zeros);
 vec xaxaeigval;
 eig_sym(xaxaeigval,xaxa);
 xaxa-=(xaxaeigval(0)-1)*eye(xaxa.n_rows,xaxa.n_cols);
 xa=chol(xaxa);
}
