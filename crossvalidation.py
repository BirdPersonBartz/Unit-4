import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

#clean the data
loansData['Interest.Rate'] = [float(interest[0:-1])/100 for interest in loansData['Interest.Rate']]
loansData['Loan.Length'] = [int(length[0:-7]) for length in loansData['Loan.Length']]
loansData['FICO.Score'] = [int(val.split('-')[0]) for val in loansData['FICO.Range']]

#creating variables out of the columns
intrate = loansData['Interest.Rate']
laonamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']


# #placing into the matrix

# x1 = np.matrix(fico).transpose()
# x2 = np.matrix(laonamt).transpose()

# #stacking the values of x1 next to x2
# x = np.column_stack([x1,x2])

# X = sm.add_constant(x)
# model = sm.OLS(y,X).fit()



x_vars = np.empty([0,0])
for i, row in loansData.iterrows():
	x_vars.append([row['Interest.Rate'], row['Loan.Length']])

y_var = loansData['FICO.Score']
elementnumber = len(loansData['FICO.Score'])

print(type(x_vars))
print(type(y_var))

# kf = KFold(elementnumber, n_folds = 9)
# for train, test in kf:
# 	x_train, x_test = x_vars[train], x_vars[test]
# 	y_train, y_test = y_var[train], y_var[test]

# print('ok')

