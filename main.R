#' Starts the whole simulation and creates all graphical outputs
#' 
#' parameters necessary for Monte Carlo simulation
#' @param n_simulations number of simulations for the Monte Carlo study
#' 
#' parameters necessary for data generating processes
#' @param n_covariates number of covariates
#'
#' parameters necessary for DML estimator
#' @param k_folds folds used for cross-fitting


# Preliminaries -----------------------------------------------------------

## Load necessary packages, set working directory and seed, remove previously stored variables

toload <- c("grf", "tidyverse", "hdm", "glmnet", "nnls", "Matrix", "matrixStats")
toinstall <- toload[which(toload %in% installed.packages()[,1] == F)]
lapply(toinstall, install.packages, character.only = TRUE)
lapply(toload, require, character.only = TRUE)

directory_path <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(directory_path)

rm(list = ls())
set.seed(123)

## Load source functions
# DGPs
source("DGP/DGP1.R")

# DML estimator
source("nonparam_DML/DML_estimator.R")

# Ensemble learner
source("ensemble_method/ensemble.R")
source("ensemble_method/ml_wrapper.R")
source("ensemble_method/utils_ensemble.R")


# Parameters --------------------------------------------------------------

### Define necessary parameters
## Monte Carlo Simulation
n_simulations = 50                  # Number of simulation rounds for Monte Carlo Study

## Data
n_covariates = 5                    # Number of confounders
n_observations = 2000               # Number of observations in simulated dataset
effect = 0.5                        # True value for effect
beta = seq(1, n_covariates, 1)/10   # Coefficients for confounders in DGP

## Ensemble method
cv_folds = 5                        # Number of folds for cross-validation of used ML methods in the ensemble method

## Double ML estimator
k_folds = 2                         # cross-fitting folds for DML estimation


# Simulation 1: Linear Case -----------------------------------------------

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
  fold_mat = prep_cf_mat(nrow(X), k_folds)
  
  # cross-fitting folds
  for (i in 1:k_folds) {
    # split the data set into main and auxiliary 
    fold = as.logical(fold_mat[,i])
    
    X_main <- X[!fold, ]
    X_aux <- X[fold, ]
    
    Y_main <- Y[!fold, ]
    Y_aux <- Y[fold, ]
    
    D_main <- D[!fold, ]
    D_aux <- D[fold, ]
    
    ### Step 1 DML: Estimate nuisance parameters
    # The ensemble needs to be trained on one partition of the data set. Then the predictions are made using the other partition
    # Using the complementary datasets ensures the cross-fitted condition put up by Chernozhukov et al. (2018)
    
    ## Ensemble for the outcome
    # estimate the conditional expectation of E[Y|X] aka the conditional outcome function
    G_ensemble_aux = regression_forest(X = X_aux, Y = Y_aux)
    G_main = as.matrix(predict(g_model_aux, newdata = as.matrix(X_fold)))
    
    G_ensemble_main = regression_forest(X = X_fold, Y = Y_fold)
    G_aux = as.matrix(predict(g_model_fold, newdata = X_aux))
    
    ## Ensemble for the p-score
    # estimate the conditional expectation of E[D|X] aka the propensity score function
    M_ensemble_aux = regression_forest(X = X_aux, Y = D_aux)
    M_main = as.matrix(predict(m_model_aux, newdata = X_fold))
    
    M_ensemble_main = regression_forest(X = X_fold, Y = D_fold)
    M_aux = as.matrix(predict(m_model_fold, newdata = X_aux))
    
    ### Step 2 DML: Derive the true effect (theta) by applying Neyman orthogonality theorem
    
    ## Calculate the residuals from the nuisance predictions, which are necessary for the orthogonality conditions
    V_aux = D_aux - m_aux
    V_fold = D_fold - m_fold
    
    # regress the residuals to get orthogonal scores
    theta_aux = dml_est(Y_aux, G_aux, V_aux)        # with models trained on fold
    theta_fold = dml_est(Y_main, G_main, V_main)    # with models trained on aux
    theta_vec[i] = mean(theta_aux, theta_fold)
    
  }
  
  theta[j] = mean(theta_vec)
}

est_effect = mean(theta)
print(est_effect)


