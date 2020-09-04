#' This function calculates the average treatment effects for all combinations.
#' Requires input from \code{PO_dmlmt}.
#'
#' @param mu Matrix mu with the individual values of the effcicient score from \code{PO_dmlmt}
#' @param cs_i If not NULL, boolean vector to indicate that observation is on support
#' @param cl If not NULL, vector with cluster variables
#' @param print If TRUE, print results
#'
#' @return \code{TE_dmlmt} returns the results average reatment effects.
#' @export

TE_dmlmt <- function(mu,cs_i=NULL,print=TRUE,cl=NULL) {
  
  # Retrieve important info
  n <- nrow(mu)
  num_t <- ncol(mu)
  nm_t <- sprintf("T%d",0:(num_t-1))
  if (is.null(cs_i)) cs_i <- rep(TRUE,n)
  
  # Initialize results
  res <- matrix(NA,sum(1:(num_t-1)),4)
  colnames(res) <- c("TE","SE","t","p")
  rownames(res) <- rep("Platzhalter",nrow(res))
  
  pos <- 1
  for (i in 1:(num_t-1)) {
    loc <- i+1
    for (j in loc:(num_t)) {
      eif_i <- mu[cs_i,j] - mu[cs_i,i]
      res[pos,1] <- mean(eif_i)
      if (is.null(cl)) {
        res[pos,2] <- sqrt(mean((eif_i-mean(eif_i))^2) / sum(cs_i))
      } else {
        res[pos,2] <-  sqrt(sum(tapply(eif_i - mean(eif_i), cl[cs_i], sum)^2) / sum(cs_i)^2)
      }
      
      rownames(res)[pos] <- paste0(nm_t[j]," - ",nm_t[i])
      pos <- pos + 1
    }
  }
  
  # t-stat
  res[,3] <- res[,1] / res[,2]
  # p-value
  res[,4] <- 2 * stats::pt(abs(res[,3]),n,lower = FALSE )
  
  if (isTRUE(print)) {
    cat("\n Average effects\n")
    stats::printCoefmat(res,has.Pvalue = TRUE)
    cat("\n# of obs on / off support:",toString(sum(cs_i))," / ",toString(sum(!cs_i)) ,"\n")
  }
  return(res)
}