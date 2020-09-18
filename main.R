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
n_simulations = 5                   # Number of simulation rounds for Monte Carlo Study

## Data
n_covariates = 5                    # Number of confounders
n_observations = 2000               # Number of observations in simulated dataset
effect = 0.5                        # True value for effect
beta = seq(1, n_covariates, 1)/10   # Coefficients for confounders in DGP

## Ensemble method
cv_folds = 2                        # Number of folds for cross-validation of used ML methods in the ensemble method

## Double ML estimator
k_folds = 2                         # cross-fitting folds for DML estimation


# Setup ensemble ----------------------------------------------------------
# Components of ensemble for the p-score
mean_ps = create_method("mean",name="Mean ps")
lasso_bin_ps = create_method("lasso",name="Lasso ps",args=list(family = "binomial"))
forest_ps =  create_method("forest_grf",name="Forest ps",args=list(tune.parameters = "all",honesty=FALSE))

# Components of ensemble for the outcome
ols_oc = create_method("ols",name="OLS oc")
lasso_oc = create_method("lasso",name="Lasso oc",args=list(family = "binomial"))
forest_oc =  create_method("forest_grf",name="Forest oc",args=list(tune.parameters = "all",honesty=FALSE))

ps_methods = list(mean_ps, forest_ps)
oc_methods = list(ols_oc, forest_oc)

# Simulation 1: Linear Case -----------------------------------------------

# create folds for cross-fitting

theta_cf = rep(NA, k_folds)
theta = rep(NA, n_simulations)

oc_ensemble_cf = matrix(NA, k_folds, length(oc_methods))
ps_ensemble_cf = matrix(NA, k_folds, length(ps_methods)) 
oc_ensemble = matrix(NA, n_simulations, length(oc_methods))
ps_ensemble = matrix(NA, n_simulations, length(ps_methods)) 

for (j in 1:n_simulations) {
  
  # simulate data
  data = DGP1(n_simulations = n_simulations,n_covariates = n_covariates, n_observations = n_observations, beta = beta, effect = effect)
  Y = data[[1]]
  D = data[[2]]
  X = data[[3]]
  n_obs = seq(1, nrow(X), 1)
  
  # construct folds
  fold_mat = prep_cf_mat(nrow(X), k_folds)
  
  # cross-fitting folds
  for (i in 1:k_folds) {
    # split the data set into main and auxiliary 
    folds = as.logical(fold_mat[,i])
    
    X_main <- X[!folds, ]
    X_aux <- X[folds, ]
    
    Y_main <- Y[!folds]
    Y_aux <- Y[folds]
    
    D_main <- D[!folds]
    D_aux <- D[folds]
    
    ### Step 1 DML: Estimate nuisance parameters
    # The ensemble needs to be trained on one partition of the data set. Then the predictions are made using the other partition
    # Using the complementary datasets ensures the cross-fitted condition put up by Chernozhukov et al. (2018)
    
    ## Ensemble for the outcome
    # estimate the conditional expectation of E[Y|X] aka the conditional outcome function
    G_ensemble_aux = ensemble(oc_methods, X_main, Y_main, nfolds=cv_folds, quiet=F, xnew=X_aux) # estimate the model
    G_aux = G_ensemble_aux$fit_full$predictions # extract predictions applying the ensemble weights
    oc_ensemble_aux = G_ensemble_aux$nnls_weights # extract the ensemble weights
    
    G_ensemble_main = ensemble(oc_methods, X_aux, Y_aux, nfolds=cv_folds, quiet=F, xnew=X_main) # estimate the model
    G_main = G_ensemble_main$fit_full$predictions # extract predictions applying the ensemble weights
    oc_ensemble_main = G_ensemble_main$nnls_weights # extract the ensemble weights
    
    oc_ensemble_cf[i, ] = colMeans(rbind(oc_ensemble_aux, oc_ensemble_main)) # store the cross-fitted average of this iteration
    
    ## Ensemble for the p-score
    # estimate the conditional expectation of E[D|X] aka the propensity score function
    M_ensemble_aux = ensemble(ps_methods, X_main, Y_main, nfolds=cv_folds, quiet=F, xnew=X_aux) # estimate the model
    M_aux = M_ensemble_aux$fit_full$predictions # extract predictions applying the ensemble weights
    ps_ensemble_aux = M_ensemble_aux$nnls_weights # extract the ensemble weights
    
    M_ensemble_main = ensemble(ps_methods, X_aux, Y_aux, nfolds=cv_folds, quiet=F, xnew=X_main) # estimate the model
    M_main = M_ensemble_main$fit_full$predictions # extract predictions applying the ensemble weights
    ps_ensemble_main = M_ensemble_main$nnls_weights # extract the ensemble weights
    
    ps_ensemble_cf[i, ] = colMeans(rbind(ps_ensemble_aux, ps_ensemble_main)) # store the cross-fitted average of this iteration
    
    ### Step 2 DML: Derive the true effect (theta) by applying Neyman orthogonality theorem
    
    ## Calculate the residuals from the nuisance predictions, which are necessary for the orthogonality conditions
    V_aux = D_aux - M_aux
    V_main = D_main - M_main
    
    # regress the residuals to get orthogonal scores
    theta_aux = dml_est(Y_aux, G_aux, V_aux)        # with models trained on fold
    theta_main = dml_est(Y_main, G_main, V_main)    # with models trained on aux
    theta_cf[i] = mean(theta_aux, theta_main)
    
  }
  
  # update list of estimates for current simulation round
  theta[j] = mean(theta_cf)                         # estimated effect theta in current simulation round
  oc_ensemble[j,] = colMeans(oc_ensemble_cf)        # weights for the ml methods in the ensemble of the function E[Y|X]
  ps_ensemble[j,] = colMeans(ps_ensemble_cf)        # weights for the ml methods in the ensemble of the function E[D|X]
  
}

est_effect = mean(theta)                            # average effect over all simulation rounds
print(est_effect)


