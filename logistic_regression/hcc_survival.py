"""
Since there're missing values in the dataset, I'll replace them with mean values.
"""

import numpy as np
import csv

DATA_FILE_PATH = 'hcc-survival/hcc-data.txt'

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
    return temp_ds
    array = np.array(temp_ds)
    return array

def print_list(_list):
    for item in _list:
        print(item)

print(ds_raw)
print(len(ds_raw))
print(ds_raw[0])
print(len(ds_raw[0]))

ds = handle_missing_values_and_convert_to_nsarray(ds_raw)
print_list(ds)

input("Press enter to continue..")
