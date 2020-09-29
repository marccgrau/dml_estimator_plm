# Averaging method (comparison purposes) ----------------------------------

mean_fit = function(x,y) {
  mean = mean(y)
  output = list("mean"=mean,"n"=nrow(x))
  output
}

predict.mean_fit = function(mean_fit,x,y,xnew=NULL,weights=FALSE) {
  
  if (is.null(xnew)) fit = rep(mean_fit$mean,nrow(x))
  else fit = rep(mean_fit$mean,nrow(xnew))
  
  if (isTRUE(weights)) w = matrix(1 / length(y),nrow(xnew),nrow(x))
  else w = NULL
  
  list("prediction"=fit,"weights"=w)
}


ols_fit = function(x,y) {
  
  x = cbind(rep(1,nrow(x)),x)
  
  ols_coef = lm.fit(x,y)$coefficients

  ols_coef
}


# Least squares regression ------------------------------------------------

predict.ols_fit = function(ols_fit,x,y,xnew=NULL,weights=FALSE) {
  
  if (is.null(xnew)) xnew = x
  
  x = cbind(rep(1,nrow(x)),x)
  xnew = cbind(rep(1,nrow(xnew)),xnew)
  
  # Remove variables that were dropped due to collinearity
  x = x[,!is.na(ols_fit)]
  xnew = xnew[,!is.na(ols_fit)]
  
  # Calculate hat matrix
  hat_mat = xnew %*% solve(crossprod(x),tol=2.225074e-308) %*% t(x)
  fit = hat_mat %*% y
  
  if (weights==FALSE) hat_mat = NULL
  
  list("prediction"=fit,"weights"=hat_mat)
}


# Cross-validated ridge regression ----------------------------------------

#' This function estimates cross-validated ridge regression based on the \code{\link{glmnet}} package
#'
#' @param x Matrix of covariates (number of observations times number of covariates matrix)
#' @param y vector of outcomes
#' @param w vector of weights
#' @param ... Pass \code{\link{glmnet}} options
#' @import glmnet
#'
#' @return An object with S3 class "glmnet"
#' @export

ridge_fit = function(x,y,args=list()) {
  
  ridge = do.call(cv.glmnet,c(list(x=x,y=y,alpha=0),args))
  
  ridge
}

predict.ridge_fit = function(ridge_fit,x,y,xnew=NULL,weights=FALSE) {
  
  if (is.null(xnew)) xnew = x
  
  fit = predict(ridge_fit,newx=xnew,type="response")
  
  if (weights==FALSE) hat_mat = NULL 
  else {
    # Get covariate matrices
    n = nrow(x)
    p = ncol(x)
    x = scale(x)
    x = cbind(rep(1,nrow(x)),x)
    xnew = scale(xnew)
    xnew = cbind(rep(1,nrow(xnew)),xnew)
    
    # Calculate hat matrix, see also (https://stats.stackexchange.com/questions/129179/why-is-glmnet-ridge-regression-giving-me-a-different-answer-than-manual-calculat)
    hat_mat = xnew %*% solve(crossprod(x) + ridge_fit$lambda.min  * n / sd(y) * diag(x = c(0, rep(1,p)))) %*% t(x)
    fit = hat_mat %*% y
  }
  
  list("prediction"=fit,"weights"=hat_mat)
}


# Cross-validated LASSO regression ----------------------------------------

#' This function estimates cross-validated lasso regression based on the \code{\link{glmnet}} package
#'
#' @param x Matrix of covariates (number of observations times number of covariates matrix)
#' @param y vector of outcomes
#' @param w vector of weights
#' @param ... Pass \code{\link{glmnet}} options
#' @import glmnet
#'
#' @return An object with S3 class "glmnet"
#' @export

lasso_fit = function(x,y,args=list()) {
  
  lasso = do.call(cv.glmnet,c(list(x=x,y=y),args))
  lasso
}

predict.lasso_fit = function(lasso_fit,x,y,xnew=NULL,weights=FALSE) {
  
  if (isTRUE(weights)) stop("No weighted representation of Lasso available.")
  if (is.null(xnew)) xnew = x
  
  fit = predict(lasso_fit,newx=xnew,type="response",s="lambda.min")
  
  list("prediction"=fit,"weights"="No weighted representation of Lasso available.")
}

# Random Forest -----------------------------------------------------------

forest_grf_fit = function(x,y,args=list()) {

  rf = do.call(regression_forest,c(list(X=x,Y=y),args))

  rf
}


predict.forest_grf_fit = function(forest_grf_fit,x,y,xnew=NULL,weights=FALSE) {
  
  if (is.null(xnew)) xnew = x
  
  fit = predict(forest_grf_fit,newdata=xnew)$prediction
  
  if (weights==TRUE) w = get_sample_weights(forest_grf_fit,newdata=xnew)
  else w = NULL
  
  list("prediction"=fit,"weights"=w)
}


# Post-LASSO regression ---------------------------------------------------

plasso_fit = function(x,y,args=list()) {
  plasso = do.call(plasso,c(list(x=x,y=y),args))
  
  plasso
}

predict.plasso_fit = function(plasso_fit,x,y,xnew=NULL,weights=FALSE) {

  if (is.null(xnew)) xnew = x
  x = add_intercept(x)
  xnew = add_intercept(xnew)
  
  # Fitted values for post lasso
  nm_act = names(coef(plasso_fit$lasso_full)[,plasso_fit$ind_min_pl])[which(coef(plasso_fit$lasso_full)[,plasso_fit$ind_min_pl] != 0)]
  
  xact = x[,nm_act,drop=F]
  xactnew = xnew[,nm_act,drop=F]
  
  # Remove potentially collinear variables
  coef = lm.fit(xact,y)$coefficients
  xact = xact[,!is.na(coef)]
  xactnew = xactnew[,!is.na(coef)]
  
  hat_mat = xactnew %*% solve(crossprod(xact),tol=2.225074e-308) %*% t(xact)
  fit_plasso = hat_mat %*% y
  if (weights==FALSE) hat_mat = NULL
  
  list("prediction"=fit_plasso,"weights"=hat_mat)
}
  


# XGBoost (eXtreme Gradient Boosting) -------------------------------------

xgboost_fit = function(x,y,args=list()) {
  
  data = xgb.DMatrix(data = x, label = y) 
  
  xgb = do.call(xgb.train, c(list(data = dx),args))
  
  xgb
}


predict.xgboost_fit = function(xgb_fit,x,y,xnew=NULL,weights=FALSE) {
  
  if (is.null(xnew)) xnew = x
  
  fit = predict(xgb_fit,newdata=xnew)

  list("prediction"=fit, "weights" = "No weighted representation of XGBoost available")
}


# Neural Network ----------------------------------------------------------

neural_net_fit = function(x,y,args=list()) {
  
  scale_x = as.data.frame(scale(x, center = TRUE, scale = TRUE))
  scale_y = as.data.frame(scale(y, center = TRUE, scale = TRUE))
  unscale_mean_y = mean(y)
  unscale_sd_y = sd(y)
  
  colnames(scale_y) = c("Y1")
  
  tempdata = cbind(scale_y, scale_x)
  
  nn_formula = as.formula(paste("Y1 ~", paste(colnames(scale_x), collapse = " + ")))
  
  NNet = do.call(neuralnet, c(list(formula = nn_formula, data = tempdata), args))
  
  list("model" = NNet, "unscale_mean" = unscale_mean_y, "unscale_sd" = unscale_sd_y)
}


predict.neural_net_fit = function(neural_net_fit,x,y,xnew=NULL,weights=FALSE) {
  
  if (is.null(xnew)) xnew = x
  
  scale_xnew = as.data.frame(scale(xnew, center = TRUE, scale = TRUE))
  
  preds = predict(neural_net_fit$model,newdata = scale_xnew)
  
  fit = (preds + neural_net_fit$unscale_mean) * neural_net_fit$unscale_sd
  
  list("prediction"=fit, "weights" = "No weighted representation of the neural network available")
}
