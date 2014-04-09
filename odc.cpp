#define MATHLIB_STANDALONE
#include <cstdlib>
#include <iostream>
#include <armadillo>
#include "Rmath.h"

using namespace std;
using namespace arma;

extern "C" void odc(double *ryo, double *rxo, int *rno, int *rns, int *rp, double *rlam, int *rniter, double *rpriorprob, double *rprobs, double *rphi, double *rys, double *rxs, double *rscale, unsigned int *rgam){


	//Define Variables//
	int acceptance=0;
	int niter=*rniter;
	int p=*rp;
	int no=*rno;
	int ns=*rns;
	double w=1;
	double a=(no-1)/2;
	double b;
	double ldensity_current=0.0;
	double ldensity_proposed=0.0;
	double gammadensity_current;
	double gammadensity_proposed;
	double phi;
	Mat<double> xs(ns,p);
	Mat<double> xa(ns,p);
	Mat<double> xc(no+ns,p);
	Mat<double> xsg;
	Mat<double> xag;
	Mat<double> xcg;
	Mat<double> xoxo(p,p);
	Mat<double> xsxs(p,p);
	Mat<double> xoxog(p,p);
	Mat<double> xsxsg(p,p);
	Mat<double> D(p,p);
	Mat<double> Lam(p,p);
	Mat<double> Lamg(p,p);
	Mat<double> xo(no,p);
	Mat<double> E(ns,ns);
	Mat<double> L(ns,ns);
	Mat<double> xog;
	Mat<double> Ino=eye(no,no);
	Mat<double> Ins=eye(ns,ns);
	Mat<double> Inc;
	Mat<double> Io;
	Mat<double> Is;
	Mat<double> P1(no,no);
	Mat<double> Px(no,no);
	Mat<double> ys_mcmc(ns,niter,fill::zeros);
	Mat<double> prob_mcmc(p,niter,fill::zeros);
	Mat<uword>  gamma_mcmc(p,niter,fill::zeros);
	Col<double> phi_mcmc(niter,fill::ones);
	Col<double> yo(no);
	Col<double> yc(no+ns);
	Col<double> one(no,fill::ones);
	Col<double> mu(ns);
	Col<double> ys(ns);
	Col<double> Z(ns);
	Col<double> lam(p);
	Col<double> d(p);
	Col<double> Bols(p);
	Col<double> xoyo(p);
	Col<double> prob(p,fill::ones);
	Col<double> priorprob(p);
	Col<double> priorodds(p);
	Col<double> odds(p);
	Col<double> ldl(p);
	Col<double> dli(p);
	Col<uword> gamma(p,fill::zeros);
	Col<uword> inc_indices(p,fill::ones);


	//Copy RData Into Matrix Classes//
	std::copy(ryo, ryo + yo.n_elem, yo.memptr());
	std::copy(rxo, rxo + xo.n_elem, xo.memptr());
	std::copy(rxs, rxs + xs.n_elem, xs.memptr());
	std::copy(rlam, rlam + lam.n_elem, lam.memptr());
	std::copy(rpriorprob, rpriorprob + priorprob.n_elem, priorprob.memptr());


	//Create Matrices//
	Io=join_rows(Ino, Ino);
	Is=join_rows(Ins,(1+w)*Ins);
	Inc=join_cols(Io, Is);
	xc=join_cols(xo,xs);
	xa=xs-xo;
	xoxo=xo.t()*xo;
	xsxs=xs.t()*xs;
	D=xsxs;
	d=D.diag();


	//Initialize Parameters at MLE//
	P1=one*(one.t()*one).i()*one.t();
	Px=xo*(xoxo).i()*xo.t();
	phi=(no-1)/dot(yo,((Ino-P1-Px)*yo));
	Bols=(xoxo).i()*xo.t()*yo;
	ys=xs*Bols;


	//Pre-Gibbs Computations Needn't Be Computed Every Iteration//
	Lam=diagmat(lam);
	for (int i = 0; i < p; ++i)
	{
		priorodds(i)=priorprob(i)/(1-priorprob(i));
		ldl(i)=sqrt(lam(i)/(d(i)+lam(i)));
		dli(i)=1/(d(i)+lam(i));
	}


	//Run Gibbs Sampler//
	ys_mcmc.col(0)=ys;
	phi_mcmc(0)=phi;
	gamma_mcmc.col(0)=gamma;
	prob_mcmc.col(0)=prob;
	xoyo=xo.t()*yo;
	for (int t = 1; t < niter; ++t)
	{
		//Form Submatrices//
		inc_indices=find(gamma);
		Lamg=Lam.submat(inc_indices,inc_indices);
		xsg=xs.cols(inc_indices);
		xog=xo.cols(inc_indices);
		xag=xa.cols(inc_indices);
		xcg=xc.cols(inc_indices);
		xoxog=xoxo.submat(inc_indices,inc_indices);
		xsxsg=xsxs.submat(inc_indices,inc_indices);

		//Draw ϕ ~ ϕ|Yo,γ//
		b=0.5*dot(yo,(Ino-P1-xog*(xoxog+Lamg).i()*xog.t())*yo);
		phi=rgamma(a,(1/b)); //rgamma uses scale

		//Draw Ys ~ Ys|Yo,γ,ϕ//
		mu=(Ins+xsg*Lamg.i()*xog.t())*solve(Ino+xog*Lamg.i()*xog.t(),yo);
		E=w*Ino+xag*(xoxog+Lamg).i()*xag.t();
		E=E/phi;
		L=chol(E);
		for (int i = 0; i < ns; ++i) Z(i)=rnorm(0,1);
		ys=mu+(L.t()*Z);

		//Calculate log[ f(Yo,Ys|γ,ϕ) ]//
		yc=join_cols(yo, ys);
		E=Inc+xcg*Lamg.i()*xcg.t();
		ldensity_current=-0.5*log(det(E))-0.5*phi*dot(yc,solve(E,yc));

		//Calculate P(γ==1|Ys,ϕ)//
		for (int i = 0; i < p; ++i)
		{
			Bols(i)=(1/d(i))*dot(xs.col(i),ys);
			odds(i)=priorodds(i)*ldl(i)*trunc_exp(0.5*phi*dli(i)*d(i)*d(i)*Bols(i)*Bols(i));
			prob(i)=odds(i)/(1+odds(i));
		}

                //Calculate  g(γ|Ys,ϕ)// 
		gammadensity_current=1;
		for (int i = 0; i < p; ++i)
		{
			if(gamma(i)==1){
				gammadensity_current*=prob(i);
			}else{
				gammadensity_current*=(1-prob(i));
			}
			
		}

		//Draw γ' ~ γ|Ys,ϕ//
		for (int i = 0; i < p; ++i)
		{
			if(runif(0,1)<prob(i)){
				gamma(i)=1;
			}else{
				gamma(i)=0;
			}
			
		}

		//Calculate  g(γ'|Ys,ϕ)// 
		gammadensity_proposed=1;
		for (int i = 0; i < p; ++i)
		{
			if(gamma(i)==1){
				gammadensity_proposed*=prob(i);
			}else{
				gammadensity_proposed*=(1-prob(i));
			}
			
		}

		//Form Submatrices//
		inc_indices=find(gamma);
		Lamg=Lam.submat(inc_indices,inc_indices);
		xsg=xs.cols(inc_indices);
		xog=xo.cols(inc_indices);
		xag=xa.cols(inc_indices);
		xcg=xc.cols(inc_indices);
		xoxog=xoxo.submat(inc_indices,inc_indices);
		xsxsg=xsxs.submat(inc_indices,inc_indices);

		//Calculate log[ f(Yo,Ys|γ',ϕ) ]
		yc=join_cols(yo, ys);
		E=Inc+xcg*Lamg.i()*xcg.t();
		ldensity_proposed=-0.5*log(det(E))-0.5*phi*dot(yc,solve(E,yc));

		//Draw γ ~ γ|Yo,Ys,ϕ via MH//
		if(log(runif(0,1))<ldensity_proposed-ldensity_current+log(gammadensity_current)-log(gammadensity_proposed)){
			//accept proposed gamma
			acceptance=acceptance+1;
		}else{
			//Do not accept proposed gamma
			gamma=gamma_mcmc.col(t-1);
		}

		//Store Draws//
		gamma_mcmc.col(t)=gamma;
		prob_mcmc.col(t)=prob;
		ys_mcmc.col(t)=ys;
		phi_mcmc(t)=phi;
	}
cout << acceptance<<endl;

	std::copy(phi_mcmc.memptr(), phi_mcmc.memptr() + phi_mcmc.n_elem, rphi);
	std::copy(prob_mcmc.memptr(), prob_mcmc.memptr() + prob_mcmc.n_elem, rprobs);
	std::copy(gamma_mcmc.memptr(), gamma_mcmc.memptr() + gamma_mcmc.n_elem, rgam);
	std::copy(ys_mcmc.memptr(), ys_mcmc.memptr() + ys_mcmc.n_elem, rys);
}
