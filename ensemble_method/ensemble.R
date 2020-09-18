
ensemble = function(ml,
                    x,y,xnew,
                    nfolds=5,
                    weights=FALSE,
                    quiet=TRUE) {

  # Matrix to store the cross-validated predictions
  fit_cv = matrix(NA,nrow(x),length(ml))
  colnames(fit_cv) = sprintf("Method%s",seq(1:length(ml)))
  for (i in 1:length(ml)) {
    if (!is.null(ml[[i]]$name)) colnames(fit_cv)[i] = ml[[i]]$name
  }
  
  # Get CV folds
  cvf = prep_cf_mat(length(y),nfolds)
  
  # Loop over different folds
  if (length(ml) > 1) {
    for (i in 1:nfolds) {
      
      # Split sample for current fold
      fold = as.logical(cvf[,i])
      x_tr = x[!fold,]
      y_tr = y[!fold]
      x_te = x[fold,]
      
      fit_cv[cvf[,i] == 1,] = ensemble_core(ml,x_tr,y_tr,x_te,quiet=quiet)$predictions
    }
    
    fit_cv[is.na(fit_cv)] = mean(y) # e.g. glmnet produces sometimes NaN for logistic Ridge
    mse_cv = colMeans((y - fit_cv)^2)
    nnls_weights = nnls(fit_cv,as.matrix(y))$x
    nnls_weights = nnls_weights / sum(nnls_weights)
    
    fit_full = ensemble_core(ml,x,y,xnew,weights=weights,quiet=quiet)
    best = fit_full$predictions[,which.min(mse_cv)]
    ensemble = fit_full$predictions %*% nnls_weights
    
    w = NULL
    if (isTRUE(weights)) {
      w = matrix(0,nrow(xnew),nrow(x))
      for (i in 1:length(ml)) {
        w = w + nnls_weights[i] * fit_full$weights[[i]]
      }
      w = Matrix(w,sparse=T)
    }
    
    colnames(fit_full$predictions) = names(mse_cv) = names(nnls_weights) = colnames(fit_cv)
  }
  else {
    fit_full = ensemble_core(ml,x,y,xnew,weights=weights,quiet=quiet)
    ensemble = best = fit_full$predictions
    
    w = nnls_weights = mse_cv = fit_cv = NULL
    if (isTRUE(weights)) w = fit_full$weights[[1]]
  }

  list("ensemble" = ensemble,"best" = best,"fit_full" = fit_full,"weights" = w,
       "nnls_weights" = nnls_weights, "mse_cv" = mse_cv, "fit_cv" = fit_cv)
}


ensemble_core = function(ml,
                         x_tr,y_tr,x_te,
                         weights=FALSE,
                         quiet=TRUE) {
  
  # Initialize objects to be filled
  fit_mat = matrix(NA,nrow(x_te),length(ml))
  weights_list = vector("list",length(ml))
  
  for (i in 1:length(ml)) {
    
    wrapper = paste0(ml[[i]]$method,"_fit")
    
    if (isFALSE(quiet)) print(wrapper)
    
    if (is.null(ml[[i]]$x_select) & length(ml[[i]]$args) == 0)          fit = do.call(wrapper,list(x=x_tr,y=y_tr))
    else if (is.null(ml[[i]]$x_select) & !(length(ml[[i]]$args) == 0))  fit = do.call(wrapper,list(x=x_tr,y=y_tr,args=ml[[i]]$args))
    else if (!is.null(ml[[i]]$x_select) & length(ml[[i]]$args) == 0)    fit = do.call(wrapper,list(x=x_tr[,ml[[i]]$x_select],y=y_tr))
    else                                                                fit = do.call(wrapper,list(x=x_tr[,ml[[i]]$x_select],y=y_tr,args=ml[[i]]$args))
    if (is.null(ml[[i]]$x_select))  temp = do.call(paste0("predict.",wrapper),list(fit,x=x_tr,y=y_tr,xnew=x_te,weights=weights))
    else                            temp = do.call(paste0("predict.",wrapper),list(fit,x=x_tr[,ml[[i]]$x_select],y=y_tr,xnew=x_te[,ml[[i]]$x_select],weights=weights))
    
    fit_mat[,i] = temp$prediction
    weights_list[[i]] = temp$weights
  }
  
  list("predictions" = fit_mat, "weights" = weights_list)
}




