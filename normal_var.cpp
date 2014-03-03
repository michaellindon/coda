#define MATHLIB_STANDALONE
#include <cstdlib>
#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

extern "C" void normal_var(double *ryo, double *rxo, int *rno, int *rp, double *rlam, int *rniter, double *rpriorprob, double *rprobs, double *rphi, double *rmu, double *rxa, double *rscale, double *rE){


	//Define Variables//
	int niter=*rniter;
	int p=*rp;
	int no=*rno;
	double a=(no-1)/2;
	double b;
	double phi;
	Mat<double> xa(p,p);
	Mat<double> xaxa(p,p);
	Mat<double> xoxo(p,p);
	Mat<double> xoxog(p,p);
	Mat<double> D(p,p);
	Mat<double> Q(p,p);
	Mat<double> Lam(p,p);
	Mat<double> xcxcLami(p,p);
	Mat<double> xo(no,p);
	Mat<double> E(p,p);
	Mat<double> L(p,p);
	Mat<double> Ino=eye(no,no);
	Mat<double> P=eye(p,p);
	Mat<double> Ip=eye(p,p);
	Mat<double> P1(no,no);
	Mat<double> Px(no,no);
	Mat<double> mu_trace(p,niter,fill::zeros);
	Mat<double> E_trace(p*p,niter,fill::zeros);
	Mat<double> prob_trace(p,niter,fill::zeros);
	Col<double> phi_trace(niter,fill::ones);
	Col<double> yo(no);
	Col<double> one(no,fill::ones);
	Col<double> mu(p,fill::zeros);
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
	std::copy(rlam, rlam + lam.n_elem, lam.memptr());
	std::copy(rpriorprob, rpriorprob + priorprob.n_elem, priorprob.memptr());


	P1=one*(one.t()*one).i()*one.t();
	//P1.fill((double)(1/no)); This doesn't work
	//Scale and Center Xo//
	for (int c = 0; c < p; ++c)
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
	xaxa.diag()=(0.001+abs(xaxa_eigenval(0)))*vec(p,fill::ones);
	xa=chol(xaxa);
	D=xaxa+xoxo;
	d=D.diag();


	//Initialize Parameters at MLE//
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
	xcxcLami=(D+Lam).i();
	for (int t = 1; t < niter; t++)
	{


		//Phi Step//
		Q=(D+Lam)*P.i();
		b=0.5*dot(yo,(Ino-P1-xo*(Q-xaxa).i()*xo.t())*yo);
		phi=((double)a)/b;

		//Ya Step//
		mu=(Ip-xa*P*xcxcLami*xa.t()).i()*xa*P*xcxcLami*xo.t()*yo;
		E=(Ip-xa*P*xcxcLami*xa.t()).i();
		E=E;

		//Probability Step//
		for (int i = 0; i < p; i++)
		{
			Bols(i)=(1/d(i))*(xoyo(i)+dot(xa.col(i),mu));
			odds(i)=priorodds(i)*ldl(i)*trunc_exp(0.5*phi*dli(i)*(d(i)*d(i)*Bols(i)*Bols(i)+dot(xa.col(i),E*xa.col(i))/phi ));
			prob(i)=odds(i)/(1+odds(i));
		}
		P.diag()=prob;

		//Store Values//
		prob_trace.col(t)=prob;
		mu_trace.col(t)=mu;
		E_trace.col(t)=vectorise(E);
		phi_trace(t)=phi;

	}


	std::copy(phi_trace.memptr(), phi_trace.memptr() + phi_trace.n_elem, rphi);
	std::copy(prob_trace.memptr(), prob_trace.memptr() + prob_trace.n_elem, rprobs);
	std::copy(mu_trace.memptr(), mu_trace.memptr() + mu_trace.n_elem, rmu);
	std::copy(E_trace.memptr(), E_trace.memptr() + E_trace.n_elem, rE);
	std::copy(xa.memptr(), xa.memptr() + xa.n_elem, rxa);
}
