"""
Since there're missing values in the dataset, I'll replace them with mean values.

20180627: Replace missing values with median, because some features are norminal.
Another moral: MinMaxScaler works significantly better than StandardScaler.
"""

import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import os
from sklearn import datasets, linear_model, discriminant_analysis#, cross_validation
from sklearn import model_selection # this replaces the cross_validation module
import pandas as pd


DATA_FILE_PATH = 'hcc-survival/hcc-data.txt'
DATA_COLUMN_FILE_PATH = 'hcc-survival/hcc-data-row-desc.txt'
TEST_DATASET_PROPORTION = 0.35
IMPORTANCE_THRESHOLD = 0.002

def read_csv(FILE_PATH):
    df = pd.read_csv(FILE_PATH, na_values='?')
    return df
    result = []
    with open(FILE_PATH, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            result.append(row)
    return result

ds_raw = read_csv(DATA_FILE_PATH)

def read_columns():
    columns = []
    with open(DATA_COLUMN_FILE_PATH, 'r') as f:
        content = f.read().strip()
        splitted = content.splitlines()
        #splitted = splitted[:-1]
        for line in splitted:
            index = line.find(': ')
            columns.append([line[:index],line[index+2:]])

    return np.array(columns)


def handle_missing_values_and_convert_to_nsarray(original_dataset):
    df = original_dataset
    columns = read_columns()
    #print(columns)
    df.columns = columns[:,0]
    cols = df.columns
    #print(df.values)
    print(df.values.shape)
    #print(df.isnull().sum())
    from sklearn.preprocessing import Imputer
    imr = Imputer(missing_values='NaN', strategy='median', axis=0)
    imr = imr.fit(df)
    imputed_data = imr.transform(df.values)
    #print(imputed_data)
    df[:] = imputed_data
    return df
    columns = len(original_dataset[0])
    rows = len(original_dataset)
    is_int = [True] * columns
    avgs = [0.0] * columns
    avg_counts = [1] * columns

    # This failed. It only created a shallow copy of the first row.
    #temp_ds = [[None]* columns] * rows

    temp_ds = [[0 for col in range(columns)] for row in range(rows)]

    for row in range(rows):
        for col in range(columns):
            value = original_dataset[row][col]
            if value == '?':
                temp_ds[row][col] = None
            else:
                fvalue = float(value)
                temp_ds[row][col] = fvalue
                is_int[col] = fvalue.is_integer()
                # calculate the mean value
                avgs[col] = avgs[col] + (fvalue - avgs[col])/avg_counts[col]
                avg_counts[col] += 1

    for row in range(rows):
        for col in range(columns):
            value = temp_ds[row][col]
            is_integer = is_int[col]
            if value != None:
                continue
            if avgs[col] < 1 and is_integer:
                if avgs[0] >= 0.5:
                    temp_ds[row][col] = 1
                else:
                    temp_ds[row][col] = 0
            elif is_integer:
                temp_ds[row][col] = int(avgs[col])
            else:
                temp_ds[row][col] = avgs[col]
    array = np.array(temp_ds)
    return array

def print_list(_list):
    for item in _list:
        print(item)

def h_theta_x(theta, x_i):
    #print(np.dot(theta, x_i))
    z = np.dot(theta, x_i)
    z = np.clip( z, -500, 500 )
    return 1/(1+np.exp(-z))

def feature_scaling_obsolete(x):
    n = x.shape[1]
    m = x.shape[0]
    feature_scales = np.array([1.0]*n)
    scaled_x = np.copy(x)
    return feature_scales, scaled_x

# I doubt if it works on matrices
#v_h_theta_x = np.vectorize(h_theta_x)

# Vectorized iteration:
#  theta = theta - alpha/m * X_transpose * (h_theta(X) - y)
def gradient_descent(alpha, X, y):
    m = len(X)
    theta_length = len(X[0])
    theta = np.zeros(theta_length)
    epsilon = 0.0000001
    iters = 0
    max_iter = 100000
    feature_scales, scaled_X = feature_scaling_obsolete(X)
    #print('scaled_X:', scaled_X)
    #print('feature_scaling: ', feature_scales)

    while True:
        hs = np.array([h_theta_x(theta, X_i) for X_i in scaled_X])
        gradient = alpha/m * np.dot(scaled_X.T, hs - y)
        theta = theta - gradient
        mag_gradient = np.dot(gradient, gradient)
        #print('mag_gradient: ', mag_gradient)
        if mag_gradient < epsilon:
            break
        iters += 1
        if iters >= max_iter:
            print('Max iteration reached!')
            break
    print('Iteration: ', iters)
    theta = theta * feature_scales

    return theta

def score(test_set, theta):
    X = test_set[:,:-1]
    y = test_set[:,-1]
    m = len(test_set)
    const = np.array([1] * m).reshape(m, 1)
    X = np.append(const, X, axis=1)
    correct = 0
    result = []
    for i in range(m):
        p = h_theta_x(theta, X[i])
        if p >= 0.5:
            h = 1.0
        else:
            h = 0.0
        result.append(h)
        if h == y[i]:
            correct += 1
    print('h: ', np.array(result))
    print('y: ', y)
    score = correct / m
    return score

def test_LogisticRegression(*data):
    print('test_LogisiticRegression:')
    X_train, X_test, y_train, y_test = data
    regr = linear_model.LogisticRegression()
    regr.fit(X_train, y_train)
    print('Coefficients: {}, intercept {:}'.format(regr.coef_, regr.intercept_))
    print('Score: {:.5}'.format(regr.score(X_test, y_test)))

def analyse_feature_importances(X_train, y_train, columns):
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=10000,
                                    random_state=0,
                                    n_jobs=-1)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    print('Feature importances:')
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f+1, 30,
                                columns[f],
                                importances[indices[f]]))
    print('Ommiting features with importance < ', IMPORTANCE_THRESHOLD)

    selected_indices = [i for i in range(X_train.shape[1]) if importances[indices[i]] > IMPORTANCE_THRESHOLD]
    X_selected = X_train[:,selected_indices]
    print('Selected features {} out of {}'.format(X_selected.shape[1], X_train.shape[1]))

    return X_selected, selected_indices


def logistic_regression(dataset):
    df = dataset
    dataset = df.values
    test_count = int(len(dataset)*TEST_DATASET_PROPORTION)
    test_set = dataset[:test_count]
    train_set = dataset[test_count:]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train_set)
    train_set = scaler.transform(train_set)
    test_set = scaler.transform(test_set)
    print('test dataset rows:', len(test_set))
    print('train dataset rows:', len(train_set))
    alpha = 0.1
    X = train_set[:,:-1]
    m = len(train_set)
    const = np.array([1] * m).reshape(m, 1)
    y = train_set[:,-1]
    X_train_selected, selected_indices = analyse_feature_importances(X, y, df.columns)
    selected_indices.append(test_set.shape[1]-1)
    test_set = test_set[:,selected_indices]
    added_X = np.append(const, X_train_selected, axis=1)

    #print(len(train_set[0]), len(X[0]))
    #print(y)
    #print(len(y))
    theta = gradient_descent(alpha, added_X, y)
    _score = score(test_set, theta)
    print('theta:', theta)
    print('score(accuracy):', _score)
    print('-'*99)
    print('Below is the built-in logistic regression model from sklearn')

    X_ds = dataset[:,:-1]
    y_ds = dataset[:,-1]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_ds, y_ds, test_size=0.35, random_state=0, stratify=y_ds)
    selected_indices = selected_indices[:-1]
    X_train = X_train[:,selected_indices]
    X_test = X_test[:,selected_indices]
    test_LogisticRegression(X_train, X_test, y_train, y_test)


#print(ds_raw)
#print(len(ds_raw))
#print(ds_raw[0])
#print(len(ds_raw[0]))

def entry():
    ds = handle_missing_values_and_convert_to_nsarray(ds_raw)
    #print_list(ds)
    print('number of records: ', len(ds))
    logistic_regression(ds)


entry()
input("Press enter to continue..")
