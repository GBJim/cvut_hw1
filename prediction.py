import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error


df = pd.read_csv("forestfires.txt", index_col=False, sep=" ")

X = df.iloc[:,0:-1].values
Y = df.iloc[:,-1].values
normalizer = Normalizer()

X = normalizer.fit_transform(X)
k_fold_cv = KFold(n=Y.shape[0], n_folds=10, shuffle=True)


sgdr = SGDRegressor()

for train_index, test_index in k_fold_cv:
	X_train, X_test = X[train_index], X[test_index]
	Y_train, Y_test = Y[train_index], Y[test_index]
	sgdr.fit(X_train, Y_train)
	pred = sgdr.predict(X_test)
	error = mean_squared_error(Y_test, pred)
	print(error)