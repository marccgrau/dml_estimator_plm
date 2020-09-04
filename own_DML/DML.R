#' Double machine learning for binary and multiple treatments'
#'
#'
#' This function estimates treatment effects for binary and multiple treatments
#' using Double Machine Learning.
#'
#' @param x Matrix of covariates (N x p matrix)
#' @param t Vector of treament indicators
#' @param y Vector of outcomes
#' @param cs If TRUE, common support will be checked
#' @param q Quantile used for enforcing common support
#' @param cl Vector with cluster variables can be provided
#' @param print If TRUE, supporting information is printed
#' @param w If TRUE, implied weights are calculated (only if pl=TRUE)
#'
#' @return \code{dmlmt} returns the results of the estimated average treatment effects and the potential outcomes. If specified, results of different SE rules and implied weights are returned.
#'
#' @export

dmlmt <- function(x,t,y,cs=TRUE,q=1,cl=NULL,print=FALSE,w=FALSE) {
  
  
  # Important parameters
  n <- length(y)
  num_t <- length(table(t))
  
  # fit propensity score
  cvfit_p <- cv.glmnet(x,t, family = "binomial",parallel=parallel,...)
  ps_mat <- predict(cvfit_p, x, s = "lambda.min", type = "response")
  ps_mat <- cbind(1-ps_mat,ps_mat)
  
  # calculate the general propensity score
  gps <- gps_cs(ps_mat,t_mat,cs=cs,q=q,print=print)
  
  # prepare matrix for outcomes
  y_mat <- matrix(NA,n,num_t)
  
  # fit the outcomes
  for (tr in 1:num_t) {
    cvfit_y <- cv.glmnet(x[t_mat[,tr]==1,],y[t_mat[,tr]==1],family = family,...)
    y_mat[,tr] <- predict(cvfit_y, x, s = "lambda.min", type = "response")
  }

  # Potential outcomes
  PO <- PO_dmlmt(t_mat,y,y_mat,gps$p,cs_i=gps$cs,cl=cl)
  # ATE
  ATE <- TE_dmlmt(PO$mu,gps$cs,cl=cl)
  
  ## Return results
  list("ATE" = ATE,"PO" = PO$results)
}

