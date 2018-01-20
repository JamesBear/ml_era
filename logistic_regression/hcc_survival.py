"""
Since there're missing values in the dataset, I'll replace them with mean values.
"""

import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import os
from sklearn import datasets, linear_model, discriminant_analysis#, cross_validation
from sklearn import model_selection # this replaces the cross_validation module


DATA_FILE_PATH = 'hcc-survival/hcc-data.txt'
TEST_DATASET_PROPORTION = 0.35

def read_csv(FILE_PATH):
    result = []
    with open(FILE_PATH, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            result.append(row)
    return result

ds_raw = read_csv(DATA_FILE_PATH)

def handle_missing_values_and_convert_to_nsarray(original_dataset):
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
    return 1/(1+np.exp(-np.dot(theta, x_i)))

def feature_scaling(x):
    n = x.shape[1]
    m = x.shape[0]
    feature_scales = np.array([1.0]*n)
    scaled_x = np.copy(x)

    for i in range(1, n):
        col_max = max(abs(x[:,i]))
        if abs(col_max) < 0.0001:
            continue
        feature_scales[i] = 1.0/col_max
        for j in range(m):
            o = scaled_x[j,i]
            scaled_x[j,i] = o*feature_scales[i]

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
    feature_scales, scaled_X = feature_scaling(X)
    print('scaled_X:', scaled_X)
    print('feature_scaling: ', feature_scales)

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



def logistic_regression(dataset):
    test_count = int(len(dataset)*TEST_DATASET_PROPORTION)
    test_set = dataset[:test_count]
    train_set = dataset[test_count:]
    print('test dataset rows:', len(test_set))
    print('train dataset rows:', len(train_set))
    alpha = 0.1
    X = train_set[:,:-1]
    m = len(train_set)
    const = np.array([1] * m).reshape(m, 1)
    added_X = np.append(const, X, axis=1)
    y = train_set[:,-1]
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
