power.spectrum <- function(t, x, omega,
                                       demeaned = FALSE, standardized = FALSE) {

    ## De-mean and standardize, if not already done. ##
    if (!demeaned) {
        x <- x - mean(x)
    }
    if (!standardized) {
        x <- x / sd(x)
    }

    tau <- tau(t, omega)

    a1 <- periodogram.component(t, x, omega, tau, "cos")
    b1 <- periodogram.component(t, x, omega, tau, "sin")

    ps <- 0.5 * (a1^2 + b1^2)

    ps

}
