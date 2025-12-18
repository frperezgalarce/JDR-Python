
Irrlyr1 <- read.delim("OGLE-BLG-RRLYR-00001.dat",sep="")
Irrlyr1
dim(Irrlyr1)

#1: tiempo, 2: magnitud, 3: error

plot(Irrlyr1[,1],Irrlyr1[,2],type="o")

Irrlyr2 <- read.delim("OGLE-BLG-RRLYR-00003.dat",sep="")
Irrlyr2 

plot(Irrlyr2[,1],Irrlyr2[,2],type="o")

## Problema 1: en cada catalogo, una misma banda, entre estrellas -> clustering
## Problema 2: comparar entre 2 catalogos, 2 bandas I'es, entre estrellas
# (primeras 100 observaciones, en caso del largo del problema). 
# Luego, ir aumentando el largo en 50

### JDR:

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

alpha = 0.5
beta = alpha*(1-alpha)
st <- Irrlyr1[,1]
delta.t = (sort(st)[length(st)] - sort(st)[1]) / length(st)
delta.f = 0.001
f.min = delta.f
f.max = 1 / (2*delta.t)
f <- seq(f.min, f.max, delta.f)
omega <- 2*pi*f

y <- Irrlyr1[,2]
sx1 <- cross.spectrum(st, y, st, y, omega) #LS periodogram
plot(sx1, type="l")

#sx2 <- cross.spec.mtls(y, y, st, omega, nw = 2) #TMT method

x <- Irrlyr1[,2]; st1 <- Irrlyr1[,1]
delta.t1 = (sort(st1)[length(st1)] - sort(st1)[1]) / length(st1)
f.max1 = 1 / (2*delta.t1); delta.f = 0.001; f.min = delta.f
f1 <- seq(f.min, f.max1, delta.f); omega1 <- 2*pi*f1

y <- Irrlyr2[,2]; st2 <- Irrlyr2[,1]
delta.t2 = (sort(st2)[length(st2)] - sort(st2)[1]) / length(st2)
f.max2 = 1 / (2*delta.t2); f2 <- seq(f.min, f.max2, delta.f); omega2 <- 2*pi*f2

sx1 <- cross.spectrum(st1, x, st1, x, omega1) #LS periodogram
plot(sx1, type="l")
Ix = integrate(cross.spectrum, lower=f.min, upper=f.max1, t1=st1, x1=x, t2=st1, x2=x)$value
plot(st1,x,type="l")

sx2 <- cross.spectrum(st2, y, st2, y, omega2) #LS periodogram
plot(sx2, type="l")
Iy = integrate(cross.spectrum, lower=f.min, upper=f.max2, t1=st2, x1=y, t2=st2, x2=y)$value
plot(st2,y,type="l")

Ixy = integrate(cross.spectrum, lower=f.min, upper=f.max1, t1=st1, x1=x, t2=st1, x2=y)$value
J[1,i] = beta*(Ix + Iy - 2*Ixy)/(2*pi)

?integrate





