#define MATHLIB_STANDALONE
#include <cstdlib>
#include <iostream>
#include <armadillo>
#include "Rmath.h"

using namespace std;
using namespace arma;

extern "C" void normal_em2(double *ryo, double *rxo, int *rno, int *rna, int *rp, double *rlam, int *rniter, double *rpriorprob, double *rprobs, double *rphi, double *rya, double *rxa, double *rscale, unsigned int *rgam){


	//Define Variables//
	int niter=*rniter;
	int p=*rp;
	int no=*rno;
	int na=*rna;
	int t=1;
	double a=(no-1)/2;
	double b;
	double delta;
	Mat<double> xa(na,p);
	Mat<double> xag;
	Mat<double> xaxa(p,p);
	Mat<double> xoxo(p,p);
	Mat<double> xoxog(p,p);
	Mat<double> D(p,p);
	Mat<double> Lam(p,p);
	Mat<double> Lamg(p,p);
	Mat<double> xo(no,p);
	Mat<double> E(na,na);
	Mat<double> xog;
	Mat<double> Ino=eye(no,no);
	Mat<double> Ina=eye(na,na);
	Mat<double> P1(no,no);
	Mat<double> ya_mcmc(na,niter,fill::zeros);
	Mat<double> prob_mcmc(p,niter,fill::zeros);
	Mat<uword>  gamma_mcmc(p,niter,fill::ones);
	Col<double> phi_mcmc(niter,fill::ones);
	Col<double> yo(no);
	Col<double> one(no,fill::ones);
	Col<double> mu(na);
	Col<double> ya(na);
	Col<double> lam(p);
	Col<double> d(p);
	Col<double> Bols(p);
	Col<double> xoyo(p);
	Col<double> prob(p,fill::ones);
	Col<double> priorprob(p);
	Col<double> priorodds(p);
	Col<double> odds(p,fill::ones);
	Col<double> ldl(p);
	Col<double> dli(p);
	Col<uword> gamma(p,fill::zeros);
	Col<uword> inc_indices(p,fill::ones);


	//Copy RData Into Matrix Classes//
	std::copy(ryo, ryo + yo.n_elem, yo.memptr());
	std::copy(rxo, rxo + xo.n_elem, xo.memptr());
	std::copy(rxa, rxa + xa.n_elem, xa.memptr());
	std::copy(rlam, rlam + lam.n_elem, lam.memptr());
	std::copy(rpriorprob, rpriorprob + priorprob.n_elem, priorprob.memptr());


	//Create Matrices//
	xoxo=xo.t()*xo;
	xaxa=xa.t()*xa;
	D=xaxa+xoxo;
	d=D.diag();
	P1=one*(one.t()*one).i()*one.t();


	//Pre-Gibbs Computations Needn't Be Computed Every Iteration//
	Lam=diagmat(lam);
	for (int i = 0; i < p; ++i)
	{
		priorodds(i)=priorprob(i)/(1-priorprob(i));
		ldl(i)=sqrt(lam(i)/(d(i)+lam(i)));
		dli(i)=1/(d(i)+lam(i));
	}


	//Run Gibbs Sampler//
	ya_mcmc.col(0)=ya;
	gamma_mcmc.col(0)=gamma;
	prob_mcmc.col(0)=prob;
	xoyo=xo.t()*yo;
	do{
		//Form Submatrices
		inc_indices=find(gamma);
		Lamg=Lam.submat(inc_indices,inc_indices);
		xag=xa.cols(inc_indices);
		xog=xo.cols(inc_indices);
		xoxog=xoxo.submat(inc_indices,inc_indices);

		//Draw Phi//
		b=0.5*dot(yo,(Ino-P1-xog*(xoxog+Lamg).i()*xog.t())*yo);

		//Draw Ya//
		mu=xag*(xoxog+Lamg).i()*xog.t()*yo;
		E=Ina+xag*(xoxog+Lamg).i()*xag.t();

		//Draw Gamma//
		for (int i = 0; i < p; ++i)
		{
			Bols(i)=(1/d(i))*(xoyo(i)+dot(xa.col(i),mu));
			odds(i)=priorodds(i)*ldl(i)*trunc_exp(0.5*(a/b)*dli(i)*d(i)*d(i)*Bols(i)*Bols(i)+0.5*dli(i)*dot(xa.col(i),E*xa.col(i)));
			prob(i)=odds(i)/(1+odds(i));
			//if(prob(i)!=prob(i)) prob(i)=1;	 //Catch NaN

			//Choose Median Probability Model
			if(0.5<prob(i)){
				gamma(i)=1;
			}else{
				gamma(i)=0;
			}
		}

		//Store Draws//
		gamma_mcmc.col(t)=gamma;
		prob_mcmc.col(t)=prob;
		ya_mcmc.col(t)=ya;
		
		delta=dot(prob_mcmc.col(t)-prob_mcmc.col(t-1),prob_mcmc.col(t)-prob_mcmc.col(t-1));
		t=t+1;
	} while(delta>0.001);

	cout << t << endl;

	std::copy(phi_mcmc.memptr(), phi_mcmc.memptr() + phi_mcmc.n_elem, rphi);
	std::copy(prob_mcmc.memptr(), prob_mcmc.memptr() + prob_mcmc.n_elem, rprobs);
	std::copy(gamma_mcmc.memptr(), gamma_mcmc.memptr() + gamma_mcmc.n_elem, rgam);
	std::copy(ya_mcmc.memptr(), ya_mcmc.memptr() + ya_mcmc.n_elem, rya);
}
