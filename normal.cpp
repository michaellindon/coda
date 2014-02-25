#include <cstdlib>
#include <iostream>
#include <armadillo>
#include <random>

using namespace std;
using namespace arma;

extern "C" void normal(double *ryo, double *rxo, int *rno, int *rp, double *rlam, int *rniter, double *rpriorprob, double *rprobs, double *rphi, double *rya, double *rxa, double *rscale, unsigned int *rgamma){


	//Define Variables//
	int niter=*rniter;
	int p=*rp;
	int no=*rno;
	int a=(no-1)/2;
	int b;
	double phi;
	Mat<double> xa(p,p);
	Mat<double> xag;
	Mat<double> xaxa(p,p);
	Mat<double> xoxo(p,p);
	Mat<double> xoxog(p,p);
	Mat<double> D(p,p);
	Mat<double> Lam(p,p);
	Mat<double> Lamg(p,p);
	Mat<double> xo(no,p);
	Mat<double> E(p,p);
	Mat<double> L(p,p);
	Mat<double> xog;
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
	Col<double> xoyo(p);
	Col<double> prob(p,fill::ones);
	Col<double> priorprob(p);
	Col<double> priorodds(p);
	Col<double> odds(p);
	Col<double> ldl(p);
	Col<double> d2dl(p);
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
		rscale[c]=sqrt(no/dot(xo.col(c),xo.col(c)));
		xo.col(c)=xo.col(c)*rscale[c];// Scale
	}

	//Create Xa//
	xoxo=xo.t()*xo;
	xaxa=(-1)*xoxo; //Force off diagonal elements of xaxa to be (-1)* off diagonal elements of xoxo
	xaxa.diag()=vec(p,fill::zeros); //Set the diagonal entries of xaxa to be zero
	eig_sym(xaxa_eigenval,xaxa); //Calculate the most negative eigenvalue
	xaxa.diag()=(0.01+abs(xaxa_eigenval(0)))*vec(p,fill::ones);
	xa=chol(xaxa);
	D=xaxa+xoxo;
	d=D.diag();


	//Initialize Parameters at MLE//
	Px=xo*(xoxo).i()*xo.t();
	phi=(no-1)/dot(yo,((Ino-P1-Px)*yo));
	Bmle=(xoxo).i()*xo.t()*yo;
	ya=xa*Bmle;


	//C++11 PRNG//
	std::mt19937 engine;
	std::normal_distribution<> N(0,1);
	std::gamma_distribution<> Ga(a,1);
	std::uniform_real_distribution<> Un(0,1);


	//Pre-Gibbs Computations Needn't Be Computed Every Iteration//
	Lam=diagmat(lam);
	for (int i = 0; i < p; i++)
	{
		priorodds(i)=priorprob(i)/(1-priorprob(i));
		ldl(i)=sqrt(lam(i)/(d(i)+lam(i)));
		d2dl(i)=(d(i)*d(i))/(d(i)+lam(i));
	}


	//Run Gibbs Sampler//
	ya_mcmc.col(0)=ya;
	phi_mcmc(0)=phi;
	gamma_mcmc.col(0)=gamma;
	prob_mcmc.col(0)=prob;
	xoyo=xo.t()*yo;
	for (int t = 1; t < niter; t++)
	{
		//Form Submatrices
		inc_indices=find(gamma);
		Lamg=Lam.submat(inc_indices,inc_indices);
		xag=xa.cols(inc_indices);
		xog=xo.cols(inc_indices);
		xoxog=xoxo.submat(inc_indices,inc_indices);

		//Draw Phi//
		b=0.5*dot(yo,(Ino-P1-xog*(xoxog+Lamg).i()*xog.t())*yo);
		phi=Ga(engine)/b;

		//Draw Ya//
		mu=xag*(xoxog+Lamg).i()*xog.t()*yo;
		E=Ip+xag*(xoxog+Lamg).i()*xag.t();
		E=E/phi;
		L=chol(E);
		Z.imbue( [&]() { return N(engine); } );
		ya=mu+L.t()*Z;

		//Draw Gamma//
		for (int i = 0; i < p; i++)
		{
			Bmle(i)=(1/d(i))*(xoyo(i)+dot(xa.col(i),ya));
			odds(i)=priorodds(i)*ldl(i)*trunc_exp(0.5*phi*d2dl(i)*Bmle(i)*Bmle(i));
			prob(i)=odds(i)/(1+odds(i));
			//if(prob(i)!=prob(i)) prob(i)=1;	 //Catch NaN

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


	std::copy(phi_mcmc.memptr(), phi_mcmc.memptr() + phi_mcmc.n_elem, rphi);
	std::copy(prob_mcmc.memptr(), prob_mcmc.memptr() + prob_mcmc.n_elem, rprobs);
	std::copy(gamma_mcmc.memptr(), gamma_mcmc.memptr() + gamma_mcmc.n_elem, rgamma);
	std::copy(ya_mcmc.memptr(), ya_mcmc.memptr() + ya_mcmc.n_elem, rya);
	std::copy(xa.memptr(), xa.memptr() + xa.n_elem, rxa);
}
