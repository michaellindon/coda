#include <cstdlib>
#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

extern "C" void normal(double *ryo, double *rxo, int *rno, int *rp, double *rlam, int *rniter){

	int niter=*rniter;
	int p=*rp;
	int no=*rno;

	Mat<double> xa(p,p);
	Mat<double> xagam;
	Mat<double> xaxa(p,p);
	Mat<double> Lam(p,p);
	Mat<double> xo(no,p);
	Col<double> yo(no);
	Col<uword> gamma(p,fill::ones);
	Col<uword> inc_indices(p,fill::ones);
	Col<double> xaxa_eigenval(p);
	Col<double> lam(p);

	std::copy(ryo, ryo + yo.n_elem, yo.memptr());
	std::copy(rxo, rxo + xo.n_elem, xo.memptr());
	std::copy(rlam, rlam + lam.n_elem, lam.memptr());
	Lam=diagmat(lam);


	xaxa=xo.t()*xo;
	xaxa.diag()=vec(p,fill::zeros);
	eig_sym(xaxa_eigenval,xaxa);
	xaxa-=(xaxa_eigenval(0)-1)*eye(xaxa.n_rows,xaxa.n_cols);
	xa=chol(xaxa);



	Col<double> phi_mcmc(niter,fill::ones);
	Mat<double> ya_mcmc(niter,p,fill::zeros);
	Mat<double> gamma_mcmc(niter,p,fill::ones);

	int a=(no-1)/2;
	int b;

	for (int t = 0; t < niter; t++)
	{
		/*Run Gibbs Sampler*/
		inc_indices=find(gamma);
		xagam=xa.cols(inc_indices);
	}

}
