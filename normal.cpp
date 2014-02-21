#include <cstdlib>
#include <iostream>
#include <armadillo>
#include <random>

using namespace std;
using namespace arma;

extern "C" void normal(double *ryo, double *rxo, int *rno, int *rp, double *rlam, int *rniter, double *rpriorprob, double *rprobs, double *rphi, double *rya, double *rxa){


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
	Mat<double> xoxo(p,p);
	Mat<double> xoxogam(p,p);
	Mat<double> D(p,p);
	Mat<double> Lam(p,p);
	Mat<double> Lamgam(p,p);
	Mat<double> xo(no,p);
	Mat<double> E(p,p);
	Mat<double> L(p,p);
	Mat<double> xogam;
	Mat<double> Ino=eye(no,no);
	Mat<double> Ip=eye(p,p);
	Mat<double> P1(no,no);
	Mat<double> Px(no,no);
	Mat<double> ya_mcmc(p,niter,fill::zeros);
	Mat<double> prob_mcmc(p,niter,fill::zeros);
	Mat<uword>  gamma_mcmc(p,niter,fill::ones);
	Col<double> phi_mcmc(niter,fill::ones);
	Col<double> yo(no);
	Col<double> one(no,fill::ones);
	Col<double> mu(p);
	Col<double> ya(p);
	Col<double> Z(p);
	Col<double> xaxa_eigenval(p);
	Col<double> lam(p);
	Col<double> d(p);
	Col<double> Bmle(p);
	Col<double> prob(p,fill::ones);
	Col<double> priorprob(p);
	Col<double> priorodds(p);
	Col<double> odds(p);
	Col<double> ldl(p);
	Col<double> dl(p);
	Col<uword> gamma(p,fill::ones);
	Col<uword> inc_indices(p,fill::ones);


	//Copy RData Into Matrix Classes//
	std::copy(ryo, ryo + yo.n_elem, yo.memptr());
	std::copy(rxo, rxo + xo.n_elem, xo.memptr());
	std::copy(rlam, rlam + lam.n_elem, lam.memptr());
	std::copy(rpriorprob, rpriorprob + priorprob.n_elem, priorprob.memptr());


	P1=one*(one.t()*one).i()*one.t();
	//P1.fill((double)(1/no)); This doesn't work
	//Scale and Center Xo//
	for (int c = 0; c < p; c++)
	{
		xo.col(c)=xo.col(c)-P1*xo.col(c); //Center
		xo.col(c)*=sqrt(no/dot(xo.col(c),xo.col(c))); // Scale
	}
	

	//Create Xa//
	xoxo=xo.t()*xo;
	xaxa=(-1)*xoxo;
	xaxa.diag()=vec(p,fill::zeros);
	eig_sym(xaxa_eigenval,xaxa);
	xaxa-=(xaxa_eigenval(0)-0.001)*eye(xaxa.n_rows,xaxa.n_cols);
	xa=chol(xaxa);
	D=xaxa+xoxo;
	d=D.diag();
	cout << D << endl;


	//Initialize Parameters at MLE//
	Px=xo*(xoxo).i()*xo.t();
	phi=(no-p)/dot(yo,((Ino-P1-Px)*yo));
	Bmle=(xoxo).i()*xo.t()*yo;
	ya=xa*Bmle;


	//C++11 PRNG//
	std::mt19937 engine;
	std::normal_distribution<> N(0,1);
	std::gamma_distribution<> Ga(a,1);
	std::uniform_real_distribution<> Un(0,1);


	//Pre-Gibbs Computations Needn't Be Computed Every Iteration//
	Lam=diagmat(lam);
	for (int c = 0; c < p; c++)
	{
		priorodds(c)=priorprob(c)/(1-priorprob(c));
		ldl(c)=sqrt(lam(c)/(d(c)+lam(c)));
		dl(c)=1/(d(c)+lam(c));
	}


	//Run Gibbs Sampler//
	ya_mcmc.col(0)=ya;
	phi_mcmc(0)=phi;
	gamma_mcmc.col(0)=gamma;
	prob_mcmc.col(0)=prob;
	for (int t = 1; t < niter; t++)
	{
		//Form Submatrices
		inc_indices=find(gamma);
		Lamgam=Lam.submat(inc_indices,inc_indices);
		xagam=xa.cols(inc_indices);
		xogam=xo.cols(inc_indices);
		xoxogam=xoxo.submat(inc_indices,inc_indices);
		Px=xogam*xoxogam.i()*xogam.t();

		//Draw Phi//
		b=0.5*as_scalar(yo.t()*(Ino-P1-xogam*(xoxogam+Lamgam).i()*xogam.t())*yo);
		phi=Ga(engine)/b;

		//Draw Ya//
		mu=xagam*(xoxogam+Lamgam).i()*xogam.t()*yo;
		E=Ip+xagam*(xoxogam+Lamgam).i()*xagam.t();
		L=chol(E);
		Z.imbue( [&]() { return N(engine); } );
		ya=mu+L*Z;

		//Draw Gamma//
		for (int i = 0; i < p; i++)
		{
			odds(i)=priorodds(i)*ldl(i)*exp(as_scalar(0.5*phi*dl(i)*(yo.t()*xo.col(i)+ya.t()*xa.col(i))*(yo.t()*xo.col(i)+ya.t()*xa.col(i))));
			prob(i)=odds(i)/(1+odds(i));
			if(prob(i)!=prob(i)) prob(i)=1;	 //Catch NaN

			if(Un(engine)<prob(i)){
				gamma(i)=1;
			}else{
				gamma(i)=0;
			}
		}

		//Store Draws//
		gamma_mcmc.col(t)=gamma;
		prob_mcmc.col(t)=prob;
		ya_mcmc.col(t)=ya;
		phi_mcmc(t)=phi;
	}

	       for (int i = 0; i < p; ++i)
		       {
			              cout <<  mean(prob_mcmc.row(i)) << endl;
			       }

	std::copy(phi_mcmc.memptr(), phi_mcmc.memptr() + phi_mcmc.n_elem, rphi);
	std::copy(prob_mcmc.memptr(), prob_mcmc.memptr() + prob_mcmc.n_elem, rprobs);
	std::copy(ya_mcmc.memptr(), ya_mcmc.memptr() + ya_mcmc.n_elem, rya);
	std::copy(xa.memptr(), xa.memptr() + xa.n_elem, rxa);
	cout << xa << endl;
}
