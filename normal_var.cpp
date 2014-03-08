#include <cstdlib>
#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

extern "C" void normal_var(double *ryo, double *rxo, int *rno, int *rna, int *rp, double *rlam, int *rniter, double *rpriorprob, double *rprobs, double *rphi, double *rmu, double *rxa, double *rscale, double *rH, double *rb){


	//Define Variables//
	int niter=*rniter;
	int p=*rp;
	int no=*rno;
	int na=*rna;
	double a=(no-1)/2;
	double b;
	double phi;
	double varyo;
	Mat<double> xa(na,p);
	Mat<double> xaxa(p,p);
	Mat<double> xoxo(p,p);
	Mat<double> xoxog(p,p);
	Mat<double> D(p,p);
	Mat<double> Q(p,p);
	Mat<double> Lam(p,p);
	Mat<double> xcxcLami(p,p);
	Mat<double> xo(no,p);
	Mat<double> H(na,na);
	Mat<double> Ino=eye(no,no);
	Mat<double> P=eye(p,p);
	Mat<double> Pi=eye(p,p);
	Mat<double> Ina=eye(na,na);
	Mat<double> P1(no,no);
	Mat<double> Px(no,no);
	Mat<double> mu_trace(na,niter,fill::zeros);
	Mat<double> H_trace(na*na,niter,fill::zeros);
	Mat<double> prob_trace(p,niter,fill::zeros);
	Col<double> phi_trace(niter,fill::ones);
	Col<double> b_trace(niter);
	Col<double> yo(no);
	Col<double> one(no,fill::ones);
	Col<double> mu(na,fill::zeros);
	Col<double> xaxa_eigenval(p);
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


	//Initialize Parameters at MLE//
	P1=one*(one.t()*one).i()*one.t();
	Px=xo*(xoxo).i()*xo.t();
	phi=(no-1)/dot(yo,((Ino-P1-Px)*yo));


	//Single Instance Computations//
	Lam=diagmat(lam);
	for (int i = 0; i < p; ++i)
	{
		priorodds(i)=priorprob(i)/(1-priorprob(i));
		ldl(i)=sqrt(lam(i)/(d(i)+lam(i)));
		dli(i)=1/(d(i)+lam(i));
	}


	//Run Variational//
	xoyo=xo.t()*yo;
	xcxcLami=diagmat(1/(d+lam));
	varyo=dot(yo,(Ino-P1)*yo);
	for (int t = 1; t < niter; t++)
	{

		//Probability Step//
		for (int i = 0; i < p; i++)
		{
			Bols(i)=(1/d(i))*(xoyo(i)+dot(xa.col(i),mu));
			odds(i)=priorodds(i)*ldl(i)*trunc_exp(0.5*phi*dli(i)*(d(i)*d(i)*Bols(i)*Bols(i)+dot(xa.col(i),solve(H,xa.col(i)))/phi ));
			prob(i)=odds(i)/(1+odds(i));
		}
		P.diag()=prob;
		Pi.diag()=1/prob;

		//Phi Step//
		Q=(D+Lam)*Pi;
		b=0.5*(varyo-dot(xoyo,solve(Q-xaxa,xoyo)));
		phi=((double)a)/b;

		//Ya Step//
		mu=solve(Ina-xa*P*xcxcLami*xa.t(),xa*P*xcxcLami*xoyo);
		H=Ina-xa*P*xcxcLami*xa.t();

		//Store Values//
		prob_trace.col(t)=prob;
		mu_trace.col(t)=mu;
		H_trace.col(t)=vectorise(H);
		phi_trace(t)=phi;
		b_trace(t)=b;

	}


	std::copy(phi_trace.memptr(), phi_trace.memptr() + phi_trace.n_elem, rphi);
	std::copy(b_trace.memptr(), b_trace.memptr() + b_trace.n_elem, rb);
	std::copy(prob_trace.memptr(), prob_trace.memptr() + prob_trace.n_elem, rprobs);
	std::copy(mu_trace.memptr(), mu_trace.memptr() + mu_trace.n_elem, rmu);
	std::copy(H_trace.memptr(), H_trace.memptr() + H_trace.n_elem, rH);
}
