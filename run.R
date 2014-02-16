rm(list=ls())
dyn.load("normal.so")
set.seed(1)
no=200
foo=rnorm(no,0,1)
sd=2
xo=cbind(foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd))
b=c(1,0,1,0,1,1,0)
xo%*%b
yo=xo%*%b+rnorm(no,0,1)
p=length(b)
niter=10000
lam=rep(1.11,p)
priorprob=rep(0.5,p)
res=.C("normal",as.double(yo),as.double(xo),as.integer(no),as.integer(p),as.double(lam),as.integer(niter),as.double(priorprob))
mlm=lm(yo~xo -1)
(1/summary(mlm)$sigma)^2
