rm(list=ls())
source("oda.bma.r")
dyn.load("normal.so")

set.seed(1)
no=200
foo=rnorm(no,0,1)
sd=0.5
xo=cbind(foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd))
b=rep(0,8);
b[1]=1;
b[5]=1;
xo%*%b
yo=xo%*%b+rnorm(no,0,3)+10000
p=length(b)
niter=20000
lam=rep(1,p)
priorprob=rep(0.5,p)
incprob=matrix(0,p,niter);
gamma=matrix(0,p,niter);
ya=matrix(0,p,niter);
B=matrix(0,p,niter);
lammcmc=matrix(0,p,niter);
phi=rep(0,niter)
xa=matrix(0,p,p);
scale=rep(0,p);




#altaltres=.C("altaltnormal",as.double(yo),as.double(xo),as.integer(no),as.integer(p),as.double(lam),as.integer(niter),as.double(priorprob),as.double(incprob),as.double(phi),as.double(ya),as.double(xa),as.double(scale),as.integer(gamma))

#C++oda
Sys.time()->start;
res=.C("normal",as.double(yo),as.double(xo),as.integer(no),as.integer(p),as.double(lam),as.integer(niter),as.double(priorprob),as.double(incprob),as.double(phi),as.double(ya),as.double(xa),as.double(scale),as.integer(gamma))
print(Sys.time()-start);
phi=(res[[9]])
prob_mcmc=as.vector(res[[8]])
prob_mcmc=matrix(prob_mcmc,p,niter)
#altaltprob_mcmc=as.vector(altaltres[[8]])
#altaltprob_mcmc=matrix(altaltprob_mcmc,p,niter)
incprob=apply(prob_mcmc[,-c(1:500)],1,mean)
#altaltincprob=apply(altaltprob_mcmc[,-c(1:500)],1,mean)
ya=as.vector(res[[10]])
ya=matrix(ya,p,niter)
#yaaltalt=as.vector(altaltres[[10]])
#yaaltalt=matrix(yaaltalt,p,niter)
xa=as.vector(res[[11]])
xa=matrix(xa,p,p)
#xaaltalt=as.vector(altaltres[[11]])
#xaaltalt=matrix(xaaltalt,p,p)
scale=res[[12]]


#Roda
simdata = data.frame(xo,yo)
burnin.sim <- 500;
Gtot <- 2000;
Sys.time()->start;
oda.n <- oda.bma(x=simdata[,-dim(simdata)[2]],y=simdata$yo,niter=Gtot,burnin=burnin.sim,model="lm",prior="normal") # Normal prior for lm 
print(Sys.time()-start);

#Enumerate
tf <- c(TRUE, FALSE)
models <- expand.grid(replicate(p,tf,simplify=FALSE))
names(models) <- NULL
models=as.matrix(models)
lmargcalc.final <- function(gammanew,xo,yo,lam)
{
  xo <- as.matrix(xo); yo <- as.vector(yo);
  no <- nrow(xo);  p <- ncol(xo);
  sqrtvar <- sqrt(apply(xo,2,var))*sqrt((no-1)/no);
  xo <- scale(xo,center=T,scale=sqrtvar);
  ncolumn <- sum(gammanew == 1);
  if (ncolumn == 0)
  {
    sigmaogamma <- diag(no);
    sigmainv <- solve(sigmaogamma);
    onetsigmainv <- apply(sigmainv,2,sum);
    a <-sum(onetsigmainv);
    siginvyo <- sigmainv%*%yo;
    b <- sum(siginvyo);
    logmarg = -.5*log(det(sigmaogamma))-.5*log(a)-((no-1)/2)*log(t(yo)%*%siginvyo-(b^2/a));
    xbeta <- rep(mean(yo),no);
  } else
  {
    Xgam <- xo[,gammanew==1];
    model.lm <- lm(yo~Xgam);
    varbeta <- 1/lam;
    betamle <- matrix(model.lm$coefficients[-1],ncolumn,1);
    alphamle <- model.lm$coefficients[1];
    betapostmean <- solve(t(Xgam)%*%Xgam+diag(ncolumn)/varbeta[gammanew==1])%*%(t(Xgam)%*%Xgam)%*%betamle
    xbeta <- Xgam%*% betapostmean+rep(alphamle,no)
    sigmaogamma <- Xgam%*%(varbeta[gammanew==1]*diag(ncolumn))%*%t(Xgam) + diag(no);
    sigmainv <- solve(sigmaogamma);
    onetsigmainv <- apply(sigmainv,2,sum);
    a <-sum(onetsigmainv);
    siginvyo <-  sigmainv%*%yo;
    b <- sum(siginvyo);
    logmarg = -.5*log(det(sigmaogamma))-.5*log(a)-((no-1)/2)*log(t(yo)%*%siginvyo-(b^2/a))    ;
  }
  #return(list(logmarg=logmarg,xbeta=xbeta));
  return((logmarg));
}
lmargcalc.my <- function(gammanew,xo,yo,lam)
{
  xo <- as.matrix(xo); yo <- as.vector(yo);
  no <- nrow(xo);  p <- ncol(xo);
  sqrtvar <- sqrt(apply(xo,2,var))*sqrt((no-1)/no);
  xo <- scale(xo,center=T,scale=sqrtvar);
  Ino=diag(no);
  one=rep(1,no);
  P1=matrix(1/no,no,no);
  Lam=diag(lam);
  ncolumn <- sum(gammanew == 1);
  a=(no+1)/2
  if (ncolumn == 0)
  {
  b=t(yo)%*%(Ino-P1)%*%yo;
    logmarg = -(0.5*(no-1))*log(2*pi)+(lgamma(a)-a*log(b)) ;
  } else
  {
    xog <- xo[,gammanew==1];
    Lamg=Lam[gammanew==1,gammanew==1];
  b=t(yo)%*%(Ino-P1-xog%*%solve(t(xog)%*%xog+Lamg)%*%t(xog))%*%yo;
    logmarg = -(0.5*(no-1))*log(2*pi)+lgamma(a) -a*log(b)+0.5*log(det(as.matrix(Lamg))) -0.5*log(det(as.matrix(Lamg+t(xog)%*%xog)))   ;
  }
  #return(list(logmarg=logmarg,xbeta=xbeta));
  return((logmarg));
}
lml.all=apply(models,1,lmargcalc.my,xo,yo,rep(1,dim(xo)[2]))
results=cbind(lml.all, models)
order=sort(results[,1],index=TRUE,decreasing=TRUE)
results[order$ix,]
results[order$ix,1]=results[order$ix,1]-results[order$ix[1],1]
results[order$ix,1]=exp(results[order$ix,1])
nconstant=sum(results[,1])
results[,1]=results[,1]/nconstant
postprob=results[order$ix,]  
round(postprob[1:10,],3)
inclusionprob=rep(0,dim(postprob)[2]-1)
for(i in 1:dim(postprob)[2]-1){
  inclusionprob[i]=sum(postprob[,i+1]*postprob[,1]);
}
round(inclusionprob,4)
round(incprob,4)
round(oda.n$incprob.rb,4)




par(mfrow=c(2,4))
for(i in 1:p){
plot(density(ya[i,]),xlim=c(min(mean(ya[i,])-3*sd(ya[1,]),mean(yaaltalt[i,])-3*sd(yaaltalt[i,])),max(mean(ya[i,])+3*sd(ya[1,]),mean(yaaltalt[i,])+3*sd(yaaltalt[i,]))))
lines(density(yaalt[i,]),col="red")
lines(density(yaaltalt[i,]),col="green")
}



foo=list()
foobar=list()
for(i in 1:p){
foo[[i]] <- hist(prob_mcmc[i,],breaks=35)
foobar[[i]] <- hist(altaltprob_mcmc[i,],breaks=35)
}



par(mfrow=c(2,4))
for(i in 1:p){
plot(foo[[i]], col="red")
plot(foobar[[i]],col="blue",add=T)
}







alpha=1;
simdata = data.frame(xo,yo)
burnin.sim <- 500;
Gtot <- 10000;
dyn.load("t_gibbs.so")
oda.t <- oda.bma(x=simdata[,-dim(simdata)[2]],y=simdata$yo,niter=Gtot,burnin=burnin.sim,model="lm",prior="Students-t",alpha=4) # Normal prior for lm 
res=.C("t_gibbs",as.double(yo),as.double(xo),as.integer(no),as.integer(p),as.double(lam),as.integer(niter),as.double(priorprob),as.double(incprob),as.double(phi),as.double(ya),as.double(xa),as.double(scale),as.integer(gamma),as.double(alpha),as.double(B),as.double(lammcmc))
prob_mcmc=as.vector(res[[8]])
prob_mcmc=matrix(prob_mcmc,p,niter)
incprob=apply(prob_mcmc[,-c(1:500)],1,mean)
incprob
oda.t$incprob.rb
lam_mcmc=as.vector(res[[16]])
lam_mcmc=matrix(lam_mcmc,p,niter)
B_mcmc=as.vector(res[[15]])
B_mcmc=matrix(B_mcmc,p,niter)
apply(B_mcmc,1,mean)
apply(oda.t$B,2,mean)
apply(lam_mcmc,1,mean)
apply(oda.t$lam,2,mean)


mean(res[[9]])
mean(oda.t$phi)





###Variational checks
niter=100
b=rep(0,niter)
lam=rep(1,p)
priorprob=rep(0.5,p)
vincprob=matrix(0,p,niter);
mu=matrix(0,p,niter);
phi=rep(0,niter)
xa=matrix(0,p,p);
scale=rep(0,p);
E=matrix(0,p*p,niter)
dyn.load("normal_var.so")

Sys.time()->start;
var=.C("normal_var",as.double(yo),as.double(xo),as.integer(no),as.integer(p),as.double(lam),as.integer(niter),as.double(priorprob),as.double(vincprob),as.double(phi),as.double(mu),as.double(xa),as.double(scale),as.double(E),as.double(b))
print(Sys.time()-start);
mu=matrix(as.vector(var[[10]]),p,niter)
mu=mu[,niter]
E=matrix(as.vector(var[[13]]),p*p,niter)
E=matrix(E[,niter],p,p)
phi=as.vector(var[[9]])
phi=phi[niter]
b=as.vector(var[[14]])
b=b[niter]
vincprob=as.vector(var[[8]])
vincprob=matrix(vincprob,p,niter)
vincprob=vincprob[,niter]

par(mfrow=c(2,4))
for(i in 1:p){
plot(density(ya[i,]))
lines(density(rnorm(10000,mu[i],sqrt(E[i,i]/phi))),col="red")
}

plot(density(res[[9]]))
lines(density(rgamma(10000,0.5*(no-1),b)),col="red")
