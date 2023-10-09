from datetime import datetime, date, timedelta
import numpy as np
import configparser
import os
import statistics
import sys
from matplotlib import colors
import matplotlib
from numpy import quantile, where, random
from scipy.stats import skew 
from sklearn import svm
# import constants as constants 
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from collections import defaultdict
import matplotlib.font_manager
import math
from sklearn import svm

# converts #of days into an actual date, in the format of (mm-dd-yyyy)
def get_date_from_line(line):
    # Splitting the line into elements
    elements = line.strip().split(',')

    if len(elements) >= 2:
        # Assuming the first element is the year
        year = elements[0]

        # Assuming the second element is the day number
        day_num = elements[1]

        # Adjusting day num
        day_num = day_num.rjust(3 + len(day_num), '0')

        # Initializing start date
        start_date = date(int(year), 1, 1)

        # Converting to date
        res_date = start_date + timedelta(days=int(day_num) - 1)
        res = res_date.strftime("%m-%d-%Y")

        # Updating the second element in the line
        elements[1] = res

        # Joining the elements back into a string
        return ','.join(elements)
    else:
        # Return the original line if it doesn't have at least two elements
        return line

# gets date and puts in a new file
def process_file_update_date(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Calling get_date_from_line to process each line
            processed_line = get_date_from_line(line)
            
            # Writing the modified line to the output file
            outfile.write(processed_line + '\n')

# aggregates the bz component of a 5min text file into hourly intervals, and also includes the date, finally putting it into a new file
def add_5th_element(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()

        for i in range(0, len(lines), 12):
            sum_of_fifth_elements = 0.0

            # Extracting the first two elements of the first line in the set
            if i < len(lines):
                first_line_elements = lines[i].strip().split(',')[:2]
                outfile.write(','.join(first_line_elements) + ',')

            for j in range(i, min(i + 12, len(lines))):
                elements = lines[j].strip().split(',')

                if len(elements) >= 5:
                    # Assuming the 5th element is a number
                    try:
                        sum_of_fifth_elements += float(elements[4])
                    except ValueError:
                        # Handle invalid values (e.g., non-numeric strings)
                        print(f"Invalid value for 5th element: {elements[4]}")

            # Rounding to the hundredth place
            sum_of_fifth_elements_rounded = round(sum_of_fifth_elements, 2)

            # Writing the rounded sum of 5th elements to the output file
            outfile.write(str(sum_of_fifth_elements_rounded) + '\n')

# removes third element of hour files as the hour increment is already known in order to facilitate comparing to 5min sum files 
def remove_third_elements_inplace(file_path):
    with open(file_path, 'r+') as file:
        lines = file.readlines()
        file.seek(0)
        file.truncate()

        for line in lines:
            # Splitting the line into elements
            elements = line.strip().split(',')

            # Removing the third element if it exists
            if len(elements) >= 3:
                del elements[2]

            # Writing the modified line back to the file
            file.write(','.join(elements) + '\n')

# gse coordinate system conversion to gsm not implemented, program assumes bz components of gse and gsm are the same
'''
#switch coordinate systems
def gse_z_to_gsm(z_gse, x_gse, y_gse):
    # Calculate beta
    beta = np.arctan2(y_gse, x_gse)
    
    # Z-coordinate transformation
    z_gsm = z_gse * np.cos(beta) - x_gse * np.sin(beta)
    
    return z_gsm
'''

# ensures both the 5min sum text file and the hour text file start at same date, and counts the total number of times the dates are the same
def find_common_start_and_count(file1_path, file2_path):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        lines_file1 = file1.readlines()
        lines_file2 = file2.readlines()

    common_start_found = False

    for i in range(len(lines_file1)):
        for j in range(len(lines_file2)):
            elements_file1 = lines_file1[i].strip().split(',')[:2]
            elements_file2 = lines_file2[j].strip().split(',')[:2]

            if elements_file1 == elements_file2:
                common_start_found = True
                print(f"Common start: {','.join(elements_file1)}")
                print(f"Line in {file1_path}: {lines_file1[i].strip()}")
                print(f"Line in {file2_path}: {lines_file2[j].strip()}")
                
                # Update both files to start with the common start
                lines_file1 = lines_file1[i:]
                lines_file2 = lines_file2[j:]
                with open(file1_path, 'w') as updated_file1:
                    updated_file1.writelines(lines_file1)
                with open(file2_path, 'w') as updated_file2:
                    updated_file2.writelines(lines_file2)

                break

        if common_start_found:
            break

    if not common_start_found:
        print("Error: No common start found.")

    # Count the number of times the first and second elements are the same for both files
    count_same_elements = sum(1 for line1, line2 in zip(lines_file1, lines_file2) if line1.strip().split(',')[:2] == line2.strip().split(',')[:2])
    print(f"Total number of common dates: {count_same_elements}")
    return count_same_elements

# represents opposite polarities of the IMF and Earth's magnetic field, opposite signs = opposite polarities
def compare_and_count_opposite_signs(data_set_name, file1_path, file2_path):
    print("compare_and_count_opposite_signs: file1_path = " + file1_path + " file2_path = " + file2_path)
    counter = 0
    gse_gsm_file = open("gse_gsm.csv", 'w')
    gse_gsm_file.write('gse,gsm')

    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        lines_file1 = file1.readlines()
        lines_file2 = file2.readlines()

    for line1, line2 in zip(lines_file1, lines_file2):
        elements_file1 = line1.strip().split(',')
        elements_file2 = line2.strip().split(',')

        if len(elements_file1) >= 3 and len(elements_file2) >= 3:
            value1 = float(elements_file1[2])
            value2 = float(elements_file2[2])

            # Check if the values have opposite signs
            if (value1 < 0 and value2 > 0) or (value1 > 0 and value2 < 0):
                gse_gsm_file.writelines('\n' + str(math.log(math.fabs(value1))) + "," + str(math.log(math.fabs(value2))))
                counter += 1
    gse_gsm_file.close()
    ocsvm(data_set_name, "gse_gsm.csv")
    return counter

# counts the number of days a magnetic reconnection occurs at least once, used to calculate average magnetic reconnection per day
def compare_and_count_opposite_signs2(file1_path, file2_path):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        lines_file1 = file1.readlines()
        lines_file2 = file2.readlines()

    seen_days_with_same_second_element = set()
    count_opposite_signs = 0

    for line1, line2 in zip(lines_file1, lines_file2):
        elements_file1 = line1.strip().split(',')
        elements_file2 = line2.strip().split(',')

        # Assuming the second element is the date
        date_string = elements_file1[1]

        # Convert date string to datetime object
        date_object = datetime.strptime(date_string, "%m-%d-%Y")

        # Assuming the third element is the number for comparing opposite signs
        third_element_file1 = float(elements_file1[2])
        third_element_file2 = float(elements_file2[2])

        # Check if the signs are opposite
        if (third_element_file1 < 0 and third_element_file2 > 0) or (third_element_file1 > 0 and third_element_file2 < 0):
            # Check if the second element is unique and the same for both files
            if date_object not in seen_days_with_same_second_element and elements_file1[1] == elements_file2[1]:
                count_opposite_signs += 1
                seen_days_with_same_second_element.add(date_object)

    print(f"Total count of unique days with opposite signs and the same second element: {count_opposite_signs}")
    return count_opposite_signs

# rounding function, specifies how many decimals float number should contain
def f2(x):
    return "{:.2f}".format(x)

# credit to: https://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html#sphx-glr-auto-examples-svm-plot-oneclass-py
# uses AI to draw an anomaly graph 
def get_min_max (X):

    x_min = np.amin(X[:, 0])
    x_max = np.amax(X[:, 0])
    y_min = np.amin(X[:, 1])
    y_max = np.amax(X[:, 1])

    # print('util.get_min_max: = ',' x_min = ', f4(x_min), ' x_max = ', f4(x_max), ' y_min = ', f4(y_min), 'y_max  = ', f4(y_max))  
    return x_min, x_max, y_min, y_max

def set_x_y_min_max(x_min, x_max, y_min, y_max):
    X_MIN_F = 0.85
    X_MAX_F = 1.25
    Y_MIN_F = 0.5
    Y_MAX_F = 1.25
    # X_MIN_F, X_MAX_F, Y_MIN_F, Y_MAX_F = constants.get_x_y_min_max_constants()
    if (x_max > 0):
        x_max *= X_MAX_F
    else:
        x_max /= X_MAX_F
    if (y_max > 0):
        y_max *= Y_MAX_F
    else:
        y_max /= Y_MAX_F
    if (x_min < 0.0):
        x_min /= X_MIN_F
    else:
        x_min *= X_MIN_F
    if (y_min < 0.0):
        y_min /= Y_MIN_F
    else:
        y_min *= Y_MIN_F

    return x_min, x_max, y_min, y_max

def ocsvm (data_set_name, data_file):
    df = pd.read_csv(data_file, low_memory=False)
    attributes = ['gse', 'gsm']

    X_train = df[attributes].to_numpy()
        
    x_min, x_max, y_min, y_max = get_min_max(X_train)
    x_min, x_max, y_min, y_max = set_x_y_min_max(x_min, x_max, y_min, y_max)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    # fit the model
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    train_nom_index = where(y_pred_train == 1)
    train_anom_index = where(y_pred_train == -1)

    train_anom_values = X_train[train_anom_index]
    train_norm_values = X_train[train_nom_index]
    X_norm = train_norm_values
    X_anom = train_anom_values
    n_anom = y_pred_train[y_pred_train == -1].size
    n_norm = y_pred_train[y_pred_train == 1].size
    anom_ratio = 100.0 * n_anom / (n_norm + n_anom)

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(12, 8))
    plt.title(data_set_name + ": Magnetic Reconnection Anomaly Detection: anomaly ratio = " + str(f2(anom_ratio)) + "%")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=1, colors="white")
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors="palevioletred")
    
    # after plt.conyour and before plt.scatter
    cs = plt.contourf(xx, yy, Z, levels=np.linspace(np.min(Z), np.max(Z), 7), cmap=plt.cm.PuBu)
    plt.colorbar(cs)
    
    s = 10
    b1 = plt.scatter(X_norm[:, 0], X_norm[:, 1], c="red", s=s, edgecolors="k")
    c = plt.scatter(X_anom[:, 0], X_anom[:, 1], c='gold', s=s, edgecolors='k') 
    plt.axis("tight")
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    
    # a = plt.contour(xx, yy, Z, levels=[0], linewidths=1, colors='orange')
    
    # plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
    plt.legend(
        [a.collections[0], b1, c],
        [
            "",
            "normal observations",
            "abnormal observations",
        ],
        loc="upper right",
        prop=matplotlib.font_manager.FontProperties(size=11),
    )

    plt.xlabel(
        "log(abs(bsn_bz))"
    )
    plt.ylabel(
        "log(abs(earth_bz))"
    )
    plt.pause(5)
    plt.show(block=False)
    # plt.show()
