#####################
### Preliminaries ###
#####################

#load/install packages
toload <- c("psych", "hdm", "ggplot2", "grf")
toinstall <- toload[which(toload %in% installed.packages()[,1] == F)]
lapply(toinstall, install.packages, character.only = TRUE)
lapply(toload, require, character.only = TRUE)
directory_path <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(directory_path)
set.seed(123)

# load relevant functions
source("DML.R")
source("te_est.R")
source("po_est.R")
source("ps_est.R")

################################
### Data, Parameters & Model ###
################################

# get example data
data(pension)

# define parameters
Y = pension$tw; D = pension$p401

# model definition to work with
X = model.matrix(~ -1 + i2 + i3 + i4 + i5 + i6 + i7 + a2 + a3 + a4 + a5 +
                   fsize + hs + smcol + col + marr + twoearn + db + pira + hown, data = pension)

###########################
### Nuisance Parameters ### Necessary to adapt for Ensemble Learners
###########################

# Initialize nuisance matrices
## amount of unique and sorted treatment values
values <- sort(unique(D))
## matrix rows defined by observations, columns defined by uniques of treatments
ps_mat <- t_mat <- y_mat <- matrix(NA,length(Y),length(values))

# Get nuisance parameter predictions
for (tr in 1:length(values)){
  t_mat[,tr] <- as.numeric(D == values[tr])
  rf_p <- regression_forest(X,t_mat[,tr])
  ps_mat[,tr] <- predict(rf_p, X)$predictions
  rf_y <- regression_forest(X[t_mat[,tr] == 1,],Y[t_mat[,tr] == 1])
  y_mat[,tr] <- predict(rf_y, X)$predictions
}

########################
### Propensity Score ###
########################

# Calculate generalized p-score and enforce common support
rf_gps <- gps_cs(ps_mat,t_mat)

#########################
### Potential Outcome ###
#########################

rf_PO <- PO_dmlmt(t_mat,Y,y_mat,rf_gps$p,cs_i=rf_gps$cs)

###############################
### Double Machine Learning ###
###############################

# Estimation of average treatment effect
rf_ATE <- TE_dmlmt(rf_PO$mu,rf_gps$cs)


