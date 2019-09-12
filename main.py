"Main file for the data mine"
import feature_extractor as fex
import numpy as np

#p = fex.outlierScore()
#p.main_detection('./single/',[1.927,7984.64, 9.964],t_range=(7940,8040))
#fex.outliers_plot(p.t_vec, p.obs_vc,p.magl,p.y_abs)
p2 = fex.timeFeat()
p2.time_feature('./single/',[1.927,7984.64, 9.964])
#p2.time_feature('./single2/',[ 0.1592,7879.01,  55.26])
p2.time_feature('./bin1/',[4.366,7884.99, 6.197])

p2.plot_score()
print(p2.counts)