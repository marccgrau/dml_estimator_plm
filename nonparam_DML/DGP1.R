#' Data Generating Process I
#' 
#' parameters necessary for Monte Carlo simulation
#' @param n_simulations number of simulations for the Monte Carlo study
#' 
#' parameters necessary for data generating processes
#' @param n_covariates number of covariates
#'

g_func = function(X){
  sin(X)^2
}

m_func = function(X, nu = 0, gamma = 1) {
  0.5/pi*(sinh(gamma)) / (cosh(gamma) - cos(X - nu))
}

DGP1 = function(n_simulations, n_covariates, n_observations, beta, effect) {
  
  # construct the correlation matrix
  init_matrix = matrix( rnorm(n_covariates*n_covariates,mean=0,sd=1), n_covariates, n_covariates)
  cov_matrix = init_matrix %*% t(init_matrix)
  corr_matrix = t(chol(cov_matrix))
  
  # simulate data
  random.normal = matrix(rnorm(n_covariates*n_observations,1,2), nrow=n_covariates, ncol=n_observations)
  init_X = corr_matrix %*% random.normal
  X = t(init_X)
  
  # noise
  epsilon = rnorm(n_observations, mean = 0, sd = 1)
  eta = rnorm(n_observations, mean = 0, sd = 1)
  
  # construct nuisance functions
  M = m_func(X %*% beta)
  G = g_func(X %*% beta)
  
  # construct treatments
  D = M + epsilon
  
  # construct outcomes 
  Y = effect * D + G + eta
  
  return(list(Y, D, X))
}

