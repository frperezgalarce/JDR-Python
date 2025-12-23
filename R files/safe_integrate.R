safe_integrate <- function(fun, lower, upper, rel.tol=1e-8, abs.tol=1e-8,
                           subdivisions=2000, stop.on.error=FALSE,
                           retry=2, fallback_grid=TRUE, grid_n=20001) {
  attempt <- function(rt, at, sub) {
    wtxt <- character(0)
    val <- withCallingHandlers(
      try(integrate(fun, lower=lower, upper=upper,
                    rel.tol=rt, abs.tol=at, subdivisions=sub,
                    stop.on.error=stop.on.error),
          silent=TRUE),
      warning = function(w) {
        wtxt <<- c(wtxt, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )
    list(res=val, warnings=wtxt, rt=rt, at=at, sub=sub)
  }

  rt <- rel.tol; at <- abs.tol; sub <- subdivisions
  out <- attempt(rt, at, sub)
  k <- 0
  while (k < retry &&
         (inherits(out$res, "try-error") || !is.list(out$res) || !is.finite(out$res$value))) {
    k <- k + 1
    rt <- max(rt, 10^(-8 + k))   # relax: 1e-8 -> 1e-7 -> 1e-6
    at <- max(at, 10^(-8 + k))
    sub <- max(sub, 2000 * (10^k))
    out <- attempt(rt, at, sub)
  }

  if (!inherits(out$res, "try-error") && is.list(out$res) && is.finite(out$res$value)) {
    if (length(out$warnings)) {
      cat("Warnings during integrate():\n")
      cat(paste0(" - ", unique(out$warnings), collapse="\n"), "\n")
    }
    return(out$res$value)
  }

  ## Fallback: trapezoid on dense grid (deterministic and often stable)
  if (fallback_grid) {
    fgrid <- seq(lower, upper, length.out=grid_n)
    ygrid <- fun(fgrid, ...)
    ## ygrid must be numeric vector
    if (any(!is.finite(ygrid))) stop("Fallback grid produced non-finite values.")
    dx <- (upper - lower) / (grid_n - 1)
    trap <- dx * (0.5*ygrid[1] + sum(ygrid[2:(grid_n-1)]) + 0.5*ygrid[grid_n])
    cat("integrate() failed; used trapezoid fallback on grid.\n")
    return(trap)
  }
  stop("safe_integrate(): integrate() failed and fallback_grid=FALSE.")
}