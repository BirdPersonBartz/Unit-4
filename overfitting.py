import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error

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

# #lin test
# poly_test1 = smf.ols(formula='y ~ 1 + X', data=test_df).fit()
# # Quad test
# poly_test2 = smf.ols(formula='y ~ 1 + X + I(X**2)', data=test_df).fit()

test = test_df['y']




lin = poly_train1.predict()
predict_lin_df = pd.DataFrame(lin)


quad = poly_train2.predict()
predict_quad_df = pd.DataFrame(quad)


print(mean_squared_error(test,lin))


# print(test_df)
# print(predict_lin_df)
# print(predict_quad_df)




