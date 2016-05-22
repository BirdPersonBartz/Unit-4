import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Set seed for reproducible results
np.random.seed(414)

# Gen toy data
X = np.linspace(0, 15, 1000)
y = 3 * np.sin(X) + np.random.normal(1 + X, .2, 1000)

train_X, train_y = X[:700], y[:700]
test_X, test_y = X[700:], y[700:]

train_df = pd.DataFrame({'X': train_X, 'y': train_y})
test_df = pd.DataFrame({'X': test_X, 'y': test_y})

# Linear Fit
poly_train1 = smf.ols(formula='y ~ 1 + X', data=train_df).fit()

# Quadratic Fit
poly_train2 = smf.ols(formula='y ~ 1 + X + I(X**2)', data=train_df).fit()

#lin test
poly_test1 = smf.ols(formula='y ~ 1 + X', data=test_df).fit()
# Quad test
poly_test2 = smf.ols(formula='y ~ 1 + X + I(X**2)', data=test_df).fit()





print("mse_model")
print("Lin train: ", poly_train1.mse_model)
print("Quad train: ", poly_train2.mse_model)
print("Lin test: ", poly_test1.mse_model)
print("Quad test: ", poly_test2.mse_model)


print("mse_residual")
print("Lin train: ", poly_train1.mse_resid)
print("Quad train: ", poly_train2.mse_resid)
print("Lin test: ", poly_test1.mse_resid)
print("Quad test: ", poly_test2.mse_resid)
