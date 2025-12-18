
cross.spectrum <- function(t1, x1, t2, x2, omega,
                                       demeaned = FALSE, standardized = FALSE) {

    ## De-mean and standardize, if not already done. ##
    if (!demeaned) {
        x1 <- x1 - mean(x1)
        x2 <- x2 - mean(x2)
    }
    if (!standardized) {
        x1 <- x1 / sd(x1)
        x2 <- x2 / sd(x2)
    }

    tau1 <- tau(t1, omega)
    tau2 <- tau(t2, omega)

    a1 <- periodogram.component(t1, x1, omega, tau1, "cos")
    a2 <- periodogram.component(t2, x2, omega, tau2, "cos")
    b1 <- periodogram.component(t1, x1, omega, tau1, "sin")
    b2 <- periodogram.component(t2, x2, omega, tau2, "sin")

    co.spectrum <- 0.5 * (a1*a2 + b1*b2)
    
    co.spectrum

}
