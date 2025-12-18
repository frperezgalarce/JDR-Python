
## -----------------------------------------------------------
## Load custom functions for spectral and cross-spectral analysis
## (each of these .r files defines one or more functions used below)
## -----------------------------------------------------------

source("cross.phase.spectrum.r")          # functions for cross-phase spectrum
source("cross.amplitude.spectrum.r")      # functions for cross-amplitude spectrum
source("amplitude.spectrum.r")            # functions for (auto) amplitude spectrum
source("tau.r")                           # auxiliary function(s), e.g. lag/tau calculations
source("squared.coherence.spectrum.r")    # functions for squared coherence spectrum
source("phase.spectrum.r")                # functions for phase spectrum
source("periodogram.phase.component.r")   # functions for phase component of periodogram
source("periodogram.component.r")         # functions for periodogram of a single series
source("cross.spectrum.r")                # main cross-spectrum estimator for two series
source("power.spectrum.r")                # power spectrum / auto-spectrum functions
source("cross.spec.mtls.r")               # multitaper / MTLS-based cross-spectrum estimator


## -----------------------------------------------------------
## Load and plot the first light curve (irregularly sampled)
## -----------------------------------------------------------

Irrlyr1 <- read.delim("OGLE-BLG-RRLYR-00001.dat", sep = "")
# Irrlyr1 <- read.delim(OGLE-BLG-LPV-000009.dat", sep = "")
# Irrlyr1 <- read.delim("OGLE-BLG-LPV-000018.dat", sep = "")
# Irrlyr1 <- read.delim("OGLE-BLG-LPV-000024.dat", sep = "")
# Irrlyr1 <- read.delim("OGLE-LMC-CEP-0002.dat", sep = "")
# Irrlyr1 <- read.delim("OGLE-LMC-CEP-0005.dat", sep = "")
# Irrlyr1 <- read.delim("OGLE-LMC-CEP-0008.dat", sep = "")

# Irrlyr1 <- read.delim("OGLE-LMC-CEP-0011.dat", sep = "")
# Irrlyr1 <- read.delim("OGLE-LMC-RRLYR-00002.dat", sep = "")
# Irrlyr1 <- read.delim("OGLE-LMC-RRLYR-00003.dat", sep = "")
# Irrlyr1 <- read.delim("OGLE-LMC-RRLYR-00004.dat", sep = "")
# Irrlyr1 <- read.delim("OGLE-LMC-RRLYR-00005.dat", sep = "")

# Irrlyr1 is a data frame:
#   column 1: time (e.g. HJD or MJD)
#   column 2: magnitude
#   column 3: magnitude error

plot(Irrlyr1[, 1], Irrlyr1[, 2], type = "o")
# Quick visualization of the first RR Lyrae light curve:
# x-axis: time, y-axis: magnitude, points connected with lines ("o")


## -----------------------------------------------------------
## Load and plot the second light curve
## -----------------------------------------------------------

Irrlyr2 <- read.delim("OGLE-BLG-RRLYR-00003.dat", sep = "")
# Same structure as Irrlyr1 but for a second star (or band)

plot(Irrlyr2[, 1], Irrlyr2[, 2], type = "o")
# Visual check of the second series


## -----------------------------------------------------------
## Global spectral parameters: mixture weight and frequency grid
## -----------------------------------------------------------

alpha <- 0.5
# alpha is a mixing parameter used in Jensen-type spectral divergence

beta  <- alpha * (1 - alpha)
# beta = alpha * (1 - alpha) = 0.25 for alpha = 0.5
# This factor appears in the Jensen-type divergence between spectra

st <- Irrlyr1[, 1]
# st: time vector for series 1 (first light curve)

delta.t <- (sort(st)[length(st)] - sort(st)[1]) / length(st)
# Approximate average sampling interval Δt:
#   Δt ≈ (t_max - t_min) / N
# Using sorted times to get min and max

delta.f <- 0.001
# Frequency resolution / step Δf for evaluating the spectrum

f.min <- delta.f
# Smallest frequency evaluated: we start at Δf instead of 0
# (avoid the exact zero-frequency bin)

f.max <- 1 / (2 * delta.t)
# Maximum frequency f_max ≈ 1 / (2 * Δt) (Nyquist-like approximation)

f <- seq(f.min, f.max, delta.f)
# Frequency grid: f_min, f_min + Δf, ..., f_max
# This will be used for evaluating auto- and cross-spectra

omega <- 2 * pi * f
# Angular frequency grid: ω = 2πf


## -----------------------------------------------------------
## Define the two time series and approximate sampling
## -----------------------------------------------------------

# series 1
x   <- Irrlyr1[, 2]    # magnitudes of first light curve
st1 <- Irrlyr1[, 1]    # corresponding times of first light curve

delta.t1 <- (max(st1) - min(st1)) / length(st1)
# Approximate average sampling interval for series 1
# (not used directly below, but kept for completeness / diagnostics)


# series 2
y   <- Irrlyr2[, 2]    # magnitudes of second light curve
st2 <- Irrlyr2[, 1]    # corresponding times of second light curve

delta.t2 <- (max(st2) - min(st2)) / length(st2)
# Approximate average sampling interval for series 2


## -----------------------------------------------------------
## Auto-spectra (power or cross-spectrum with itself)
## -----------------------------------------------------------

sx1 <- cross.spectrum(st1, x, st1, x, omega)
# sx1: cross-spectrum of series 1 with itself (i.e. auto-spectrum)
# arguments:
#   t1 = st1, x1 = x -> first series
#   t2 = st1, x2 = x -> same series
#   omega         -> angular frequency grid
# Result is typically complex-valued as a function of ω

Ix <- integrate(
  cross.spectrum,
  lower = f.min,
  upper = f.max,
  t1    = st1, x1 = x,
  t2    = st1, x2 = x
)$value
# Ix: integral over frequency of the auto-spectrum of series 1
#   Ix = ∫_{f_min}^{f_max} S_xx(f) df
# Numerically computed via integrate(), using cross.spectrum as integrand.


sx2 <- cross.spectrum(st2, y, st2, y, omega)
# sx2: cross-spectrum of series 2 with itself (auto-spectrum for y)

Iy <- integrate(
  cross.spectrum,
  lower = f.min,
  upper = f.max,
  t1    = st2, x1 = y,
  t2    = st2, x2 = y
)$value
# Iy: integral over frequency of the auto-spectrum of series 2
#   Iy = ∫_{f_min}^{f_max} S_yy(f) df


## -----------------------------------------------------------
## Cross term: cross-spectrum between series 1 and 2
## -----------------------------------------------------------

f.max.xy <- min(f.max, f.max)
# Upper limit for the cross integral.
# Here min(f.max, f.max) is simply f.max; this line is equivalent to:
#   f.max.xy <- f.max
# (You could simplify to f.max.xy <- f.max.)

Ixy <- integrate(
  cross.spectrum,
  lower = f.min,
  upper = f.max.xy,
  t1    = st1, x1 = x,   # first series (times st1, data x)
  t2    = st2, x2 = y    # second series (times st2, data y)
)$value
# Ixy: integral of the cross-spectrum between series 1 and 2:
#   Ixy = ∫_{f_min}^{f_max_xy} S_xy(f) df
# This captures how much shared spectral power / covariance there is
# between the two light curves over the frequency band [f_min, f_max_xy].


## -----------------------------------------------------------
## Jensen-type spectral divergence between the two series
## -----------------------------------------------------------

J <- beta * (Ix + Iy - 2 * Ixy) / (2 * pi)
# J is a scalar divergence-like measure between the two series in the
# spectral domain. The structure:
#   Ix + Iy - 2 * Ixy
# resembles:
#   ∫ [S_xx(f) + S_yy(f) - 2 S_xy(f)] df
# which is related to the energy of the difference between spectra.
# The factor beta = alpha(1 - alpha) (with alpha = 0.5) and division by 2π
# are normalization / scaling constants consistent with the chosen
# spectral definitions.
# Intuitively, higher J means the two time series have more different
# spectral content (less similarity), lower J means they are more similar.

J
# Print the final scalar spectral distance / divergence value

