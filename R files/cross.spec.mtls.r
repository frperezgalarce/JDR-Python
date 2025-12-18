# x, y The data to be analysed. x and y can be either a two-column numerical dataframe or matrix, with sampling times in columnn 1 and measurements in column 2, a single numerical vector containing measurements, or a single vector ts object (which will be converted to a numerical vector).
# nw Thompson's frequency bandwidth parameter (>= 1)
# k Number of tapers, usually 2nw or 2nw - 1 (defaults to 2 nw)
# demean remove mean from timeseries prior to spectral estimation
# ... Additional arguments passed to gplot.mtm.
# return object of class spec with the following list items:
# \item{"freq"}{A vector with spectrum frequencies}
# \item{"spec"}{A vector with spectral power estimates corresponding to "freq"}
# \item{"series"}{Name of input time series}
# \item{"method"}{Method name: "MTLS"}
# Example:
# x <- rnorm( 256 )
# s <- spec.mtls( x, 1:256, plot = FALSE )
# plot(s$freq, s$spec, type="l")

cross.spec.mtls <- function( x, y, t = NULL, omega, nw = 4, k = NULL, demean = TRUE) {

  if ( is.null( k ) ) k = 2 * nw
  if ( k < 1 ) {
    warning( "k coerced to 1" )
    k <- 1
  }

  if ( demean ) {
    x <- x - mean( x )
    y <- y - mean( y )
  }

  t <- t[!is.na( x )]
  x <- x[!is.na( x )]
  y <- y[!is.na( y )]
  
  tapers <- multitaper::dpss( length( t ), k, nw )
  tapx <- tapers$v[,1] * x
  tapy <- tapers$v[,1] * y

  ss <- cross.spectrum(t, tapx, t, tapy, omega) 

  Ixy = integrate(cross.spectrum, lower=f.min, upper=f.max, 
                 t1=t, x1=tapx, t2=t, x2=tapy)$value

  raw <- data.frame( freq = f, spec = ss)
  spec <- list( freq = f, spec = ss )
  freq <- f
  power <- ss

  if ( k > 1 ) {
    for ( i in 2:k ) {
      tapx <- tapers$v[,i] * x
      tapy <- tapers$v[,i] * y
      ss <- cross.spectrum(t, tapx, t, tapy, omega) 
	Ixy.aux = integrate(cross.spectrum, lower=f.min, upper=f.max, 
                 t1=t, x1=tapx, t2=t, x2=tapy)$value
	Ixy = Ixy + Ixy.aux
      power <- power + ss
    }
  }
  
  power <- power / k
  Ixy = Ixy / k
  spec <- list( freq = freq, spec = power, Integral = Ixy)

  return( spec )
}