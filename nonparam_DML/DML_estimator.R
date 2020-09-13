dml_est = function(Y, G, V) {
  
  # This function provides the second step of the DML estimator, where the residuals are regressed on each other.
  # The residuals are derived from the first step of the DML procedure, where the functions g(.) and m(.) are approximated. 
  # Regressing the residuals allows for the estimation of the true effect theta
  
  theta = mean(V * (Y - G)) / mean(V * V)
  
  return(theta)
}