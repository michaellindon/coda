rm(list=ls())
source("oda.bma.r")
dyn.load("normal.so")

#Generate Data
set.seed(1)
no=200
foo=rnorm(no,0,1)
sd=0.4
xo=cbind(foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd))
b=rep(0,8);
b[1]=1;
b[5]=1;
xo%*%b
yo=xo%*%b+rnorm(no,0,3)+10000

#Scale Data and Produce xa
xo=scale(xo,center=T,scale=F)
var=apply(xo^2,2,sum)
xo=scale(xo,center=F,scale=sqrt(var/no))
xoxo=t(xo)%*%xo
A=-xoxo
diag(A)=0
diag(A)=abs(min(eigen(A)$values))+0.001
xa=chol(A)
V=eigen(A)$vectors
L=eigen(A)$values
L=sqrt(L)
xa=diag(L)%*%t(V)

###Variational checks
no=length(yo)
na=dim
niter=100
b=rep(0,niter)
p=dim(xa)[2]
na=dim(xa)[1]
lam=rep(1,dim(xa)[2])
priorprob=rep(0.5,dim(xa)[2])
vincprob=matrix(0,dim(xa)[2],niter);
mu=matrix(0,dim(xa)[1],niter);
phi=rep(0,niter)
scale=rep(0,dim(xa)[2]);
E=matrix(0,dim(xa)[1]*dim(xa)[1],niter)
dyn.load("normal_var.so")

Sys.time()->start;
var=.C("normal_var",as.double(yo),as.double(xo),as.integer(length(yo)),as.integer(dim(xa)[1]),as.integer(dim(xa)[2]),as.double(lam),as.integer(niter),as.double(priorprob),as.double(vincprob),as.double(phi),as.double(mu),as.double(xa),as.double(scale),as.double(E),as.double(b))
print(Sys.time()-start);
mu=matrix(as.vector(var[[11]]),na,niter)
mu=mu[,niter]
E=matrix(as.vector(var[[14]]),na*na,niter)
E=matrix(E[,niter],na,na)
v.phi=as.vector(var[[10]])
v.phi=v.phi[niter]
b=as.vector(var[[15]])
b=b[niter]
vincprob=as.vector(var[[9]])
vincprob=matrix(vincprob,p,niter)
vincprob=vincprob[,niter]


#EM Check
dyn.load("normal_em.so")
na=dim(xa)[1]
p=dim(xa)[2]
niter=100
lam=rep(1,p)
priorprob=rep(0.5,p)
em_incprob=matrix(0,dim(xa)[2],niter)
gamma=matrix(0,dim(xa)[2],niter);
ya=matrix(0,dim(xa)[1],niter);
phi=rep(0,niter)
scale=rep(0,dim(xa)[2]);
em=.C("normal_em",as.double(yo),as.double(xo),as.integer(no),as.integer(na),as.integer(p),as.double(lam),as.integer(niter),as.double(priorprob),as.double(em_incprob),as.double(phi),as.double(ya),as.double(xa),as.double(scale))
em.ya=as.vector(em[[11]])
em.ya=matrix(em.ya,na,niter)
em.ya=em.ya[,niter]
em.incprob=as.vector(em[[9]])
em.incprob=matrix(em.incprob,p,niter)
em.incprob=em.incprob[,niter]

#Create variables to pass to C++
na=dim(xa)[1]
p=dim(xa)[2]
niter=10000
lam=rep(1,p)
priorprob=rep(0.5,p)
incprob=matrix(0,p,niter);
gamma=matrix(0,p,niter);
ya=matrix(0,na,niter);
B=matrix(0,p,niter);
lammcmc=matrix(0,p,niter);
phi=rep(0,niter)
scale=rep(0,p);

#C++oda
Sys.time()->start;
res=.C("normal",as.double(yo),as.double(xo),as.integer(no),as.integer(na),as.integer(p),as.double(lam),as.integer(niter),as.double(priorprob),as.double(incprob),as.double(phi),as.double(ya),as.double(xa),as.double(scale),as.integer(gamma))
print(Sys.time()-start);
g.phi=(res[[10]])
prob_mcmc=as.vector(res[[9]])
prob_mcmc=matrix(prob_mcmc,p,niter)
g.incprob=apply(prob_mcmc[,-c(1:500)],1,mean)
g.ya=as.vector(res[[11]])
g.ya=matrix(g.ya,na,niter)
g.scale=res[[13]]

par(mfrow=c(2,4))
for(i in 1:8){
plot(density(g.ya[i,]))
abline(v=em.ya[i],col="red")
lines(density(rnorm(10000,mu[i],sqrt(E[i,i]/v.phi))),col="green")
}


xa=xa*sqrt(1/40)
xapxp=xa
xa=rbind(xa,qr.Q(qr(rWishart(1,8,diag(8))[,,1]))%*%xapxp)
xa=rbind(xa,qr.Q(qr(rWishart(1,8,diag(8))[,,1]))%*%xapxp)
xa=rbind(xa,qr.Q(qr(rWishart(1,8,diag(8))[,,1]))%*%xapxp)
xa=rbind(xa,qr.Q(qr(rWishart(1,8,diag(8))[,,1]))%*%xapxp)
xa=rbind(xa,qr.Q(qr(rWishart(1,8,diag(8))[,,1]))%*%xapxp)
xa=rbind(xa,qr.Q(qr(rWishart(1,8,diag(8))[,,1]))%*%xapxp)
xa=rbind(xa,qr.Q(qr(rWishart(1,8,diag(8))[,,1]))%*%xapxp)
xa=rbind(xa,qr.Q(qr(rWishart(1,8,diag(8))[,,1]))%*%xapxp)
xa=rbind(xa,qr.Q(qr(rWishart(1,8,diag(8))[,,1]))%*%xapxp)
xa=rbind(xa,xa,xa,xa)

#xa=rbind(xa,(qr.Q(qr((rWishart(1,8,diag(8))[,,1]))))%*%xapxp)
#xa=rbind(xa,(qr.Q(qr((rWishart(1,8,diag(8))[,,1]))))%*%xapxp)
#xa=rbind(xa,(qr.Q(qr((rWishart(1,8,diag(8))[,,1]))))%*%xapxp)
#xa=rbind(xa,(qr.Q(qr((rWishart(1,8,diag(8))[,,1]))))%*%xapxp)
#xa=rbind(xa,(qr.Q(qr((rWishart(1,8,diag(8))[,,1]))))%*%xapxp)
#xa=rbind(xa,(qr.Q(qr((rWishart(1,8,diag(8))[,,1]))))%*%xapxp)
#xa=rbind(xa,(qr.Q(qr((rWishart(1,8,diag(8))[,,1]))))%*%xapxp)
#xa=rbind(xa,(qr.Q(qr((rWishart(1,8,diag(8))[,,1]))))%*%xapxp)
#xa=rbind(xa,(qr.Q(qr((rWishart(1,8,diag(8))[,,1]))))%*%xapxp)
#xa=rbind(xa,(qr.Q(qr((rWishart(1,8,diag(8))[,,1]))))%*%xapxp)




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







par(mfrow=c(2,4))
for(i in 1:na){
plot(density(g.ya[i,]))
lines(density(rnorm(10000,mu[i],sqrt(E[i,i]/v.phi))),col="red")
}

plot(density(res[[10]]))
lines(density(rgamma(10000,0.5*(no-1),b)),col="red")



