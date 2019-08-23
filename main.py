"Main file for the data mine"
import feature_extractor as fex
import numpy as np

p = fex.outlierScore()
p.main_detection('./single/',[1.927,7984.64, 9.964],t_range=(7940,8040))
fex.outliers_plot(p.t_vec, p.obs_vc,p.magl,p.y_abs)