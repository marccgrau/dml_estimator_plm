#' This function calculates the potential outcomes for all treatment levels.
#'
#' @param t Matrix of binary treament indicators (N x # of treatments matrix)
#'          each column contains a binary indicator for each tratment
#' @param y Vector of outcomes
#' @param y_mat N x # of treatments matrix with fitted outcome values
#' @param p_mat N x # of treatments matrix with (generalized) propensity scores
#' @param cs_i If not NULL, boolean vector to indicate that observation is on support
#' @param cl If not NULL, vector with cluster variables
#' @param print If TRUE, print results
#'
#' @return \code{PO_dmlmt} returns a list containing the \code{results} and a matrix
#'          with the individual values of the effcicient score (\code{mu}).
#' @export

PO_dmlmt <- function(t,y,y_mat,p_mat,cs_i=NULL,cl=NULL,print=TRUE) {
  
  # Retrieve important info
  n <- nrow(t)
  num_t <- ncol(t)
  if (is.null(cs_i)) cs_i = rep(TRUE,n)
  
  # Initialize matrices
  w_ipw <- matrix(0,n,num_t)
  mu_mat <- matrix(NA,n,num_t)
  res <- matrix(NA,num_t,2)
  rownames(res) <- sprintf("Treatment %d",0:(num_t-1))
  colnames(res) <- c("PO","SE")
  
  for (i in 1:num_t) {
    w_ipw[cs_i,i] <- as.matrix(t[cs_i,i] / p_mat[cs_i,i],ncol=1)
    w_ipw[cs_i,i] <- norm_w_to_n(w_ipw[cs_i,i,drop=F])
  }
  
  for (i in 1:num_t) {
    # Potential outcome ES for individual
    mu_mat[cs_i,i] <- w_ipw[cs_i,i] * (y[cs_i] - y_mat[cs_i,i])  + y_mat[cs_i,i]
  }
  
  # Calculate Mean PO
  res[,1] <- colMeans(mu_mat,na.rm=TRUE)
  
  # Calculate SE for PO
  if (is.null(cl)) {
    for (i in 1:num_t) {
      res[i,2] <- sqrt(mean((mu_mat[cs_i,i] - mean(mu_mat[cs_i,i]))^2) / sum(cs_i))
    }
  } else {
    for (i in 1:num_t) {
      res[i,2] <- sqrt(sum(tapply(mu_mat[cs_i,i] - mean(mu_mat[cs_i,i]), cl[cs_i], sum)^2) / sum(cs_i)^2)
    }
  }
  
  if (isTRUE(print)) {
    cat("\n\n Potential outcomes:\n")
    stats::printCoefmat(res)
  }
  
  list("results"=res,"mu"=mu_mat)
}

###########################################################################################################

#' Function to normalize weights to N or to N in treated and controls separately

#' @param w vector of weights that should be normalized
#' @param d vector of treament indicators
#'
#' @return Normalized weights

norm_w_to_n <- function(w,d=NULL) {
  
  if (is.null(d)) {
    ## Normalize weights to sum up to N
    w <- w / sum(w)* nrow(w)
  } else {
    # Separate weights of treated and controls
    w1 <- w * d
    w0 <- w * (1-d)
    # Normalize weights to sum to N in both groups
    w1 <- w1 / sum(w1) * nrow(w)
    w0 <- w0 / sum(w0) * nrow(w)
    # Unify weights again
    w <- w1 + w0
  }
  return(w)
}