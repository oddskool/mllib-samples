from sklearn.datasets.svmlight_format import load_svmlight_files
from sklearn.metrics.metrics import mean_squared_error
from sklearn.tree.tree import DecisionTreeRegressor

X_train, y_train, X_test, y_test = load_svmlight_files(("YearPredictionMSD", "YearPredictionMSD.t"))

clf = DecisionTreeRegressor(max_depth=5)
clf.fit(X_train.toarray(), y_train)

print clf

y_pred = clf.predict(X_test.toarray())

print "MSE", mean_squared_error(y_test, y_pred)