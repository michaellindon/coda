#include <cstdlib>
#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

extern "C" void normal(double *ryo, double *rxo, int *rno, int *rp, int *rniter){

	int niter=*rniter;
	int p=*rp;
	int no=*rno;

	mat xa(p,p);
	mat xaxa(p,p);
	mat xo(no,p);
	colvec yo(no);
	ivec gamma(p,fill::ones);
	vec xaxaeigval;

	std::copy(ryo, ryo + yo.n_elem, &yo(0));
	std::copy(rxo, rxo + xo.n_elem, &xo(0,0));

	xaxa=xo.t()*xo;
	xaxa.diag()=vec(p,fill::zeros);
	eig_sym(xaxaeigval,xaxa);
	xaxa-=(xaxaeigval(0)-1)*eye(xaxa.n_rows,xaxa.n_cols);
	xa=chol(xaxa);



	colvec phi_mcmc(niter,fill::ones);
	mat ya_mcmc(niter,p,fill::zeros);
	mat gamma_mcmc(niter,p,fill::ones);

	int a=(no-1)/2;
	int b;

	for (int t = 0; t < niter; t++)
	{
		/*Run Gibbs Sampler*/


	}

}
