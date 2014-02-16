#include <cstdlib>
#include <iostream>
#include <armadillo>
#include <random>

using namespace std;
using namespace arma;

extern "C" void normal(double *ryo, double *rxo, int *rno, int *rp, double *rlam, int *rniter){


	//Define Variables//
	int niter=*rniter;
	int p=*rp;
	int no=*rno;
	int a=(no-1)/2;
	int b;
	double phi;
	Mat<double> xa(p,p);
	Mat<double> xagam;
	Mat<double> xaxa(p,p);
	Mat<double> Lam(p,p);
	Mat<double> xo(no,p);
	Mat<double> Ino=eye(no,no);
	Mat<double> Px(no,no);
	Mat<double> ya_mcmc(niter,p,fill::zeros);
	Mat<double> gamma_mcmc(niter,p,fill::ones);
	Col<double> phi_mcmc(niter,fill::ones);
	Col<double> yo(no);
	Col<double> ya(p);
	Col<double> Z(p);
	Col<double> xaxa_eigenval(p);
	Col<double> lam(p);
	Col<double> Bmle(p);
	Col<uword> gamma(p,fill::ones);
	Col<uword> inc_indices(p,fill::ones);


	//Copy RData Into Matrix Classes//
	std::copy(ryo, ryo + yo.n_elem, yo.memptr());
	std::copy(rxo, rxo + xo.n_elem, xo.memptr());
	std::copy(rlam, rlam + lam.n_elem, lam.memptr());



	//Create Xa//
	xaxa=xo.t()*xo;
	xaxa.diag()=vec(p,fill::zeros);
	eig_sym(xaxa_eigenval,xaxa);
	xaxa-=(xaxa_eigenval(0)-1)*eye(xaxa.n_rows,xaxa.n_cols);
	xa=chol(xaxa);

	//Scale and Center Xo//
	for (int c = 0; c < p; c++)
	{
		xo.col(c)-=mean(xo.col(c))*colvec(no,fill::ones); //Center
		xo.col(c)*=sqrt(no/dot(xo.col(c),xo.col(c))); // Scale
	}



	//Initialize Parameters at MLE//
	Px=xo*(xo.t()*xo).i()*xo.t();
	phi=(no-p)/dot(yo,((Ino-Px)*yo));
	Bmle=(xo.t()*xo).i()*xo.t()*yo;
	ya=xa*Bmle;


	//Miscellaneous//
	Lam=diagmat(lam);
	yo=yo-mean(yo); //Center Yo
	std::mt19937 engine;
	std::normal_distribution<> N(0,1);


	for (int t = 0; t < niter; t++)
	{
		//Run Gibbs Sampler//
		inc_indices=find(gamma);
		xagam=xa.cols(inc_indices);
		Z.imbue( [&]() { return N(engine); } );
	}



}
