rm(list=ls())
source("oda.bma.r")
dyn.load("normal.so")
set.seed(1)
no=200
foo=rnorm(no,0,1)
sd=3
xo=cbind(foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd),foo+rnorm(no,0,sd))
b=c(1,0,1,0,1,1,0)
xo%*%b
yo=xo%*%b+rnorm(no,0,1)+10000
p=length(b)
niter=10000
lam=rep(1,p)
priorprob=rep(0.5,p)
incprob=matrix(0,p,niter);
phi=rep(0,niter)

#C++oda
Sys.time()->start;
res=.C("normal",as.double(yo),as.double(xo),as.integer(no),as.integer(p),as.double(lam),as.integer(niter),as.double(priorprob),as.double(incprob),as.double(phi))
print(Sys.time()-start);
mean(res[[9]])
incprob=as.vector(res[[8]])
incprob=matrix(incprob,p,niter)
incprob=apply(incprob,1,mean)

#Roda
simdata = data.frame(xo,yo)
burnin.sim <- 500;
Gtot <- 10000;
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
lml.all=apply(models,1,lmargcalc.final,xo,yo,rep(1,dim(xo)[2]))
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
enumerate_inc_prob=round(inclusionprob,3)
enumerate_inc_prob
