#' This function checks common support for a user provided matrix of
#' (generalized) propensity scores
#'
#' @param p_mat N x # of treatments matrix with (generalized) propensity scores
#' @param t Matrix of binary treament indicators (N x # of treatments matrix)
#'          each column contains a binary indicator for each tratment
#' @param q Quantile used for enforcing common support
#' @param print If TRUE, descriptives for p-scores and common support shown
#' @param cs If TRUE common support enforced, if FALSE only boolean vector added
#' @import psych
#' @return \code{gps_cs} returns a list containing a N x # of treatments matrix
#'          with generalized propensity scores (\code{p}) and a boolean indicating common
#'          and a boolean indicating common support (\code{cs}).
#' @export

gps_cs <- function(p_mat,t,q=1,print=TRUE,cs=TRUE) {
  # Retrieve important info
  n <- nrow(p_mat)
  num_t <- ncol(t)
  
  # Intialize matrices
  minmax <- matrix(NA,2,num_t)
  cs_mat <- matrix(NA,n,num_t)
  
  for (i in 1:num_t) {
    for (j in 1:num_t) {
      minmax[1,j] <- stats::quantile(p_mat[t[,j]==1,i],1-q)
      minmax[2,j] <- stats::quantile(p_mat[t[,j]==1,i],q)
    }
    cs_mat[,i] <- (p_mat[,i] < max(minmax[1,]) | p_mat[,i] > min(minmax[2,]))
  }
  
  cs_ind <- rep(TRUE,n)
  if (isTRUE(cs)) cs_ind <- !apply(cs_mat,1,any)
  
  if (isTRUE(print)) {
    cat("\n\nPscores\n")
    print(psych::describe(p_mat))
    
    if (isTRUE(cs)) {
      cat("\nOff support\n", toString(sum(!cs_ind)))
      cat("\n\nPscores on support\n")
      print(psych::describe(p_mat[cs_ind,]))
    }
  }
  
  list("p"=p_mat,"cs"=cs_ind)
}


