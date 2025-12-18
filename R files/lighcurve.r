#install.packages("iAR")
library(iAR)

source("cross.phase.spectrum.r")
source("cross.amplitude.spectrum.r")
source("amplitude.spectrum.r")
source("tau.r")
source("squared.coherence.spectrum.r")
source("phase.spectrum.r")
source("periodogram.phase.component.r")
source("periodogram.component.r")
source("cross.spectrum.r")
source("power.spectrum.r")
source("cross.spec.mtls.r")

## Simulation

set.seed(6714)
st<-gentime(n=100)
y<-IARsample(phi=0.99,st=st,n=100)
y<-y$series
plot(st,y)

phi=IARloglik(y=y,st=st)$phi
fit=IARfit(phi=phi,y=y,st=st)
lines(st,fit,col="red")

alpha = 0.5
beta = alpha*(1-alpha)
delta.t = (sort(st)[length(st)] - sort(st)[1]) / length(st)
delta.f = 0.001
f.min = delta.f
f.max = 1 / (2*delta.t)
f <- seq(f.min, f.max, delta.f)
omega <- 2*pi*f

sx1 <- cross.spectrum(st, y, st, y, omega)
sx2 <- cross.spec.mtls(y, y, st, omega, nw = j)

plot(omega,sx1,type="l",col="blue",lwd=2)
lines(omega,sx2$spec,type="l",col="red",lwd=2)

n=1000
st<-gentime(n=n)
delta.t = (sort(st)[length(st)] - sort(st)[1]) / length(st)
delta.f = 0.001
f.min = delta.f
f.max = 1 / (2*delta.t)
f <- seq(f.min, f.max, delta.f)
omega <- 2*pi*f

phi <- seq(0.01,0.99,0.025)
J <- matrix(NA,5,length(phi))

for(i in 1:length(phi)){
	x <- IARsample(phi=phi[i],st=st, n=n)$series
	loglik = IARloglik(y=x, st=st)$phi
	y <- IARfit(phi=loglik, y=x, st=st)

	Ix = integrate(cross.spectrum, lower=f.min, upper=f.max, t1=st, x1=x, t2=st, x2=x)$value
	Iy = integrate(cross.spectrum, lower=f.min, upper=f.max, t1=st, x1=y, t2=st, x2=y)$value
	Ixy = integrate(cross.spectrum, lower=f.min, upper=f.max, t1=st, x1=x, t2=st, x2=y)$value
	J[1,i] = beta*(Ix + Iy - 2*Ixy)/(2*pi)

	for(j in 2:5){
	sx <- cross.spec.mtls(x, x, st, omega, nw = j)
	sy <- cross.spec.mtls(y, y, st, omega, nw = j)
	sxy <- cross.spec.mtls(x, y, st, omega, nw = j)
	J[j,i] = beta*(sx$Integral + sy$Integral - 2*sxy$Integral)/(2*pi)
	}
}

plot(phi, J[1,], type="o", ylim=c(min(J),max(J)), 
ylab="JDR", xlab=expression(phi), main=expression(n~"= 1000"))
lines(phi, J[2,], type="o", col="red")
lines(phi, J[3,], type="o", col="blue")
lines(phi, J[4,], type="o", col="violet")
lines(phi, J[5,], type="o", col="gray")
legend("topright",c("LS","b=2","b=3","b=4","b=5"), lty=1,
col=c("black","red","blue","violet","gray"),bty="n",pch=19)

#?IARsample

## clcep

data(clcep)
f1=0.060033386 #0.1
m<-foldlc(clcep,f1)
x <- m$folded[,2]
time <- m$folded[,1]
phi2=IARloglik(y=x,st=time)$phi
fit2=IARfit(phi=phi2,y=x,st=time)
fit2

plot(time,x,type="p")
lines(time,fit2,col="red")

alpha = 0.5
beta = alpha*(1-alpha)
delta.t = (sort(st)[length(st)] - sort(st)[1]) / length(st)
delta.f = 0.001
f.min = delta.f
f.max = 1 / (2*delta.t)
f <- seq(f.min, f.max, delta.f)
omega <- 2*pi*f

sx1 <- cross.spectrum(time, x, time, x, omega)
sx2 <- cross.spec.mtls(x, x, time, omega, nw = 2)

plot(omega,sx1,type="l",col="blue",lwd=2)
lines(omega,sx2$spec,type="l",col="red",lwd=2)

Ix = integrate(cross.spectrum, lower=f.min, upper=f.max, t1=time, x1=x, t2=time, x2=fit2)$value
Iy = integrate(cross.spectrum, lower=f.min, upper=f.max, t1=st, x1=y, t2=st, x2=y)$value
Ixy = integrate(cross.spectrum, lower=f.min, upper=f.max, t1=st, x1=x, t2=st, x2=y)$value
beta*(Ix + Iy - 2*Ixy)/(2*pi)

## agn

data(agn)

phi3=IARloglik(y=agn$m,st=agn$t)$phi
fit3=IARfit(phi=phi3,y=agn$m,st=agn$t)
fit3

plot(agn$t,agn$m,type="l",ylab="",xlab="")
lines(agn$t,fit3,col="red")

## eb

data(eb)
f1=1.510571586
m <- foldlc(eb,f1,plot=FALSE)
x <- m$fold[,2]
time <- m$fold[,1]

phi4=IARloglik(y=x,st=time)$phi
fit4=IARfit(phi=phi4,y=x,st=time)
fit4

plot(time,x)
lines(time,fit4,col="red")


