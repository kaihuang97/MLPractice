import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report



def z_trans(data):
  return (data - data.mean()) / data.std()


def clean_data(data): 
	data.columns=['price', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'score']

	data['price'].replace(['vhigh', 'high', 'med', 'low'], [4, 3, 2, 1], inplace=True)
	data['lug_boot'].replace(['big', 'med', 'small'], [3, 2, 1], inplace=True)
	data['maint'].replace(['vhigh', 'high', 'med', 'low'], [4, 3, 2, 1], inplace=True)
	data['safety'].replace(['high', 'med', 'low'], [3, 2, 1], inplace=True)
	data['doors'].replace(['2', '3', '4', '5more'], [2, 3, 4, 5], inplace=True)
	data['persons'].replace(['2', '4', 'more'], [2, 4, 5], inplace=True)
	data['score'].replace(['unacc', 'acc', 'good', 'vgood'], [1, 2, 3, 4], inplace=True)

	data['p'] = z_trans(data['price'])
	data['m'] = z_trans(data['maint'])
	data['s'] = z_trans(data['safety'])
	data['d'] = z_trans(data['doors'])
	data['pp'] = z_trans(data['persons'])
	data['lug'] = z_trans(data['lug_boot'])
	
	return data


def preprocess(data):
	X = data[['p', 'm', 's', 'd', 'pp', 'lug']]
	y = data['score']

	return X, y


def split_data(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
	return X_train, X_test, y_train, y_test


def mlp(X_train, X_test, y_train, y_test):
	
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	clf.fit(X_train, y_train)

	y_pred_train = clf.predict(X_train)
	y_pred_test = clf.predict(X_test)

	print('Training set R^2 =', accuracy_score(y_train, y_pred_train))
	print('Test set R^2 =', accuracy_score(y_test, y_pred_test))


def test_hyperparameters(X_train, X_test, y_train, y_test):
	mlp = MLPClassifier(max_iter=100)
	parameter_space = {
	    'hidden_layer_sizes': [(3,3,3), (5,2), (3,2), (5,5), (5,6), (2,3), (3,3), (4,4), (6, 6, 6), (6, 6), (6, 2), (6, 3), (6, 4), (6, 5)],
	    'activation': ['identity', 'logistic', 'tanh', 'relu'],
	    'solver': ['sgd', 'adam', 'lbfgs'],
	    'alpha': [0.0001, 0.05, 1e-5, 0.01, 0.001],
	    'learning_rate': ['constant', 'adaptive', 'invscaling'],
	}

	clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
	clf.fit(X_train, y_train)

	# Best paramete set
	print('Best parameters found:', clf.best_params_)

	# All results
	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
	    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
	    
	y_true, y_pred = y_test , clf.predict(X_test)

	print('Results on the test set:')
	print(classification_report(y_true, y_pred))


def main():
	file_id = 'car.data'
	link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/{FILE_ID}'
	csv_url = link.format(FILE_ID = file_id)

	data = pd.read_csv(csv_url)
	data = clean_data(data)
	X, y = preprocess(data)
	X_train, X_test, y_train, y_test = split_data(X, y)
	# mlp(X_train, X_test, y_train, y_test)
	test_hyperparameters(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
	main()
