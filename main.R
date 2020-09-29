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

toload <- c("grf", "tidyverse", "hdm", "glmnet", "nnls", "Matrix", "matrixStats", "xgboost", "neuralnet")
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
source("DGP/DGP2.R")

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


# Simulation 1: Linear Case -----------------------------------------------

# Hyperparameter Tuning for DGP 1
## Data simulation for cross-validation of ml methods to select hyperparameters
data_cv = DGP2(n_simulations = n_simulations,n_covariates = n_covariates, n_observations = n_observations, beta = beta, effect = effect)
Y_cv = data_cv[[1]]
D_cv = data_cv[[2]]
X_cv = data_cv[[3]]

## Lasso hyperparameters
### Lasso hyperparameters are computationally less expensive to estimate
### Nontheless, the approximate region of the best lambda minimzing the deviance is determined before the Monte Carlo simulation

seq_lambda_test = seq(0, 100, 0.01)

### Potential outcome
lambda_cv_oc = cv.glmnet(X_cv, Y_cv, nfolds = 10, lambda = seq_lambda_test, alpha = 1)$lambda.1se

seq_lambda_final_oc = if(lambda_cv_oc - 0.05 < 0) {seq(0, lambda_cv_oc + 0.05, 0.001)} else{seq(lambda_cv_oc - 0.05, lambda_cv_oc + 0.05, 0.001)}

### Propensity Score
lambda_cv_ps = cv.glmnet(X_cv, D_cv, nfolds = 10, lambda = seq_lambda_test, alpha = 1)$lambda.1se

seq_lambda_final_ps = if(lambda_cv_ps - 0.05 < 0) {seq(0, lambda_cv_ps + 0.05, 0.001)} else{seq(lambda_cv_ps - 0.05, lambda_cv_ps + 0.05, 0.001)}

## XGBoost hyperparameters

## Baseline model
### Boosting parameters
min_child_weight = 1
max_depth = 6
colsample_bytree = 0.8
subsample = 0.8
lambda_xgb = 1
alpha_xgb = 1

### Learning paramters
eta = 0.3
eval_metric = "RMSE"
seed = 123
objective = "reg.linear"
early_stopping_rounds = 50


## create a random search algorithm 
## following the idea of: https://towardsdatascience.com/getting-to-an-hyperparameter-tuned-xgboost-model-in-no-time-a9560f8eb54b
### Create empty lists
lowest_error_list = list()
parameters_list = list()

for (iter in 1:10000){
  param_xgb <- list(booster = "gbtree",
                objective = "binary:logistic",
                max_depth = sample(3:10, 1),
                eta = runif(1, .01, .3),
                subsample = runif(1, .7, 1),
                colsample_bytree = runif(1, .6, 1),
                min_child_weight = sample(0:10, 1)
  )
  parameters_xgb <- as.data.frame(param_xgb)
  parameters_list[[iter]] <- parameters_xgb
}


# Setup the ml methods used in the ensemble for the estimation of the nuisance parameters
# ML methods used for propensity score estimation
lasso_bin_ps_1 = create_method("lasso", name = "Lasso ps 1", args = list(family = "binomial", lambda = seq_lambda_final))
xgb_ps_1 = create_method("xgboost", name = "XGBoost ps", args = list(nrounds = 500, booster = "gbtree", verbose = FALSE))
nnet_ps_1 = create_method("neural_net", name = "NeuralNet oc", args = list(hidden = c(5), linear.output = TRUE, stepmax = 20000, threshold = 0.4))

# ML methods used for potential outcome estimation
lasso_bin_oc_1 = create_method("lasso", name = "Lasso oc 1", args = list(family = "binomial"))
xgb_oc_1 = create_method("xgboost", name = "XGBoost oc", args = list(nrounds = 500, booster = "gbtree", verbose = FALSE))
nnet_oc_1 = create_method("neural_net", name = "NeuralNet oc", args = list(hidden = c(5), linear.output = TRUE, stepmax = 20000, threshold = 0.4))

# list the respective methods for each ensemble
ps_methods_1 = list(lasso_ps, xgb_ps, neural_net_ps)
oc_methods_1 = list(lasso_oc, xgb_oc, neural_net_oc)

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
    G_ensemble_aux = ensemble(oc_methods_1, X_main, Y_main, nfolds=cv_folds, quiet=F, xnew=X_aux) # estimate the model
    G_aux = G_ensemble_aux$ensemble # extract predictions applying the ensemble weights
    oc_ensemble_aux = G_ensemble_aux$nnls_weights # extract the ensemble weights
    
    G_ensemble_main = ensemble(oc_methods_1, X_aux, Y_aux, nfolds=cv_folds, quiet=F, xnew=X_main) # estimate the model
    G_main = G_ensemble_main$ensemble # extract predictions applying the ensemble weights
    oc_ensemble_main = G_ensemble_main$nnls_weights # extract the ensemble weights
    
    oc_ensemble_cf[i, ] = colMeans(rbind(oc_ensemble_aux, oc_ensemble_main)) # store the cross-fitted average of this iteration
    
    ## Ensemble for the p-score
    # estimate the conditional expectation of E[D|X] aka the propensity score function
    M_ensemble_aux = ensemble(ps_methods_1, X_main, Y_main, nfolds=cv_folds, quiet=F, xnew=X_aux) # estimate the model
    M_aux = M_ensemble_aux$ensemble # extract predictions applying the ensemble weights
    ps_ensemble_aux = M_ensemble_aux$nnls_weights # extract the ensemble weights
    
    M_ensemble_main = ensemble(ps_methods_1, X_aux, Y_aux, nfolds=cv_folds, quiet=F, xnew=X_main) # estimate the model
    M_main = M_ensemble_main$ensemble # extract predictions applying the ensemble weights
    ps_ensemble_main = M_ensemble_main$nnls_weights # extract the ensemble weights
    
    ps_ensemble_cf[i, ] = colMeans(rbind(ps_ensemble_aux, ps_ensemble_main)) # store the cross-fitted average of this iteration
    
    ### Step 2 DML: Derive the true effect (theta) by applying Neyman orthogonality theorem
    
    ## Calculate the residuals from the nuisance predictions, which are necessary for the orthogonality conditions
    V_aux = D_aux - M_aux
    V_main = D_main - M_main
    
    # regress the residuals to get orthogonal scores
    theta_aux = dml_est(Y_aux, G_aux, V_aux)        # with models trained on main for G and M
    theta_main = dml_est(Y_main, G_main, V_main)    # with models trained on aux for G and M
    theta_cf[i] = mean(theta_aux, theta_main)
    
  }
  
  # update list of estimates for current simulation round
  theta[j] = mean(theta_cf)                         # estimated effect theta in current simulation round
  oc_ensemble[j,] = colMeans(oc_ensemble_cf)        # weights for the ml methods in the ensemble of the function E[Y|X]
  ps_ensemble[j,] = colMeans(ps_ensemble_cf)        # weights for the ml methods in the ensemble of the function E[D|X]
  
}

# Averaging over all simulations
# Average treatment effect
est_effect = mean(theta)                            # average effect over all simulation rounds

# Ensemble weights of E[Y|X]
oc_ensemble_weights = as.data.frame(t(colMeans(oc_ensemble)))
for (i in 1:length(oc_methods)) {
  if (!is.null(oc_methods[[i]]$name)) colnames(oc_ensemble_weights)[i] = oc_methods[[i]]$name
  oc_ensemble = as.data.frame(oc_ensemble)
  colnames(oc_ensemble)[i] = oc_methods[[i]]$name
}


# Ensemble weights of E[D|X]
ps_ensemble_weights = as.data.frame(t(colMeans(ps_ensemble)))
for (i in 1:length(ps_methods)) {
  if (!is.null(ps_methods[[i]]$name)) colnames(ps_ensemble_weights)[i] = ps_methods[[i]]$name
  ps_ensemble = as.data.frame(ps_ensemble)
  colnames(ps_ensemble)[i] = ps_methods[[i]]$name
}

# Print the results
paste("Average treatment effect:", round(est_effect, 3))
paste(sprintf("Ensemble weight E[Y|X] %s:",colnames(oc_ensemble_weights)), round(oc_ensemble_weights, 3))
paste(sprintf("Ensemble weight E[Y|X] %s:",colnames(ps_ensemble_weights)), round(ps_ensemble_weights, 3))


