#' Non-parametric Double ML Estimator with arbitrary ML models to estimate the nuisance parameters
#'
#'
#' The DML estimator estimates the CATE and ATE for binary treatments
#' using Double Machine Learning.
#' 
#' Necessary base parameters (received from main file):
#' @param k_folds folds used for cross-fitting
#' @param n_covariates number of covariates
#' @param n_simulations number of simulations
#'
#' What we receive from the DGP (triggered by main file):
#' @param n_observations number of observations in the dataset
#' @param y (n_observations x 1) vector of length n_observations; outcome vector
#' @param t (n_observations x 1) vector of length n_observations; treatment vector (binary)
#' @param X (n_covariates X p) matrix; controls for each sample
#' 
#' Trained throughout the process:
#' @param y_hat fitted outcome
#' @param t_hat fitted treatments
#' @param fitted_res estimator for regressing the fitted outcome residuals and the fitted treatment residuals
#' 
#' Outputs
#' @param CATE
#' @param ATE
#' 
#' @return Returns the CATE and ATE for 
#' @export
#' 
#' 

## Preliminaries
#load/install packages
toload <- c("grf", "caret")
toinstall <- toload[which(toload %in% installed.packages()[,1] == F)]
lapply(toinstall, install.packages, character.only = TRUE)
lapply(toload, require, character.only = TRUE)

directory_path <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(directory_path)

set.seed(123)

# get necessary functions
source("DGP1.R")

#### This parameters and data should be coming from the main file

# define necessary parameters
k_folds = 2
n_covariates = 5
n_simulations = 50
n_observations = 2000

# true parameters of linear model
effect = 0.5
beta = seq(1, n_covariates, 1)/10


#########################################
### Estimation of nuisance parameters ###
#########################################

# get data


# create folds for cross-fitting

theta_vec = rep(NA, k_folds)
theta_test = rep(NA, k_folds)
theta = rep(NA, n_simulations)

for (j in 1:n_simulations) {
  
  # simulate data
  data = DGP1(n_simulations = n_simulations,n_covariates = n_covariates, n_observations = n_observations, beta = beta, effect = effect)
  Y = as.data.frame(data[[1]])
  D = as.data.frame(data[[2]])
  X = as.data.frame(data[[3]])
  n_obs = seq(1, nrow(X), 1)
  
  # construct folds
  folds = createFolds(n_obs, k = k_folds, list = TRUE, returnTrain = FALSE)
  
  # cross-fitting folds
  for (i in 1:k_folds) {
    X_fold <- X[folds[[i]], ]
    X_aux <- X[-folds[[i]], ]
    Y_fold <- Y[folds[[i]], ]
    Y_aux <- Y[-folds[[i]], ]
    D_fold <- D[folds[[i]], ]
    D_aux <- D[-folds[[i]], ]
    
    # estimate the conditional expectation of E[Y|X] aka the conditional outcome function
    g_model_aux = regression_forest(X = X_aux, Y = Y_aux)
    g_fold = as.matrix(predict(g_model_aux, newdata = as.matrix(X_fold)))
    
    g_model_fold = regression_forest(X = X_fold, Y = Y_fold)
    g_aux = as.matrix(predict(g_model_fold, newdata = X_aux))
    
    # estimate the conditional expectation of E[D|X] aka the propensity score function
    m_model_aux = regression_forest(X = X_aux, Y = D_aux)
    m_fold = as.matrix(predict(m_model_aux, newdata = X_fold))
    
    m_model_fold = regression_forest(X = X_fold, Y = D_fold)
    m_aux = as.matrix(predict(m_model_fold, newdata = X_aux))
    
    # derive the residuals V
    V_aux = D_aux - m_aux
    V_fold = D_fold - m_fold

    # regress the residuals to get orthogonal scores
    theta_aux = mean(V_aux * (Y_aux - g_aux)) / mean(V_aux * V_aux)        # with models trained on fold
    theta_fold = mean(V_fold * (Y_fold - g_fold)) / mean(V_fold * V_fold)    # with models trained on aux
    theta_vec[i] = mean(theta_aux, theta_fold)
    
  }
  
  theta[j] = mean(theta_vec)
}

est_effect = mean(theta)
print(est_effect)

