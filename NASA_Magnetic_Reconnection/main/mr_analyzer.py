'''
Created on Oct 7, 2023

@author: wliu
'''

import util.functions as util
import numpy as np
import os, sys
import matplotlib.pyplot as plt
# processes all relevant data from Ace Spacecraft
data_set_name = "ACE"
print("Ace:")
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_path)
data_file_1 = parent_dir + '/data/ace_5min_bsn_bz_sum.txt'
data_file_2 = parent_dir + '/data/ace_hour_earth_bz.txt'
file_path = parent_dir + '/data/ace_hour_earth_bz.txt'
# used initially to match up files
'''
util.process_file_update_date(data_file_1, data_file_2)
util.add_5th_element(data_file_1, data_file_2)
util.remove_third_elements_inplace(file_path)
'''
count_same_elements = util.find_common_start_and_count(data_file_1, data_file_2)
opposite_signs_count = util.compare_and_count_opposite_signs(data_set_name, data_file_1, data_file_2)
print(f"Number of magnetic reconnections in ace: {opposite_signs_count}") 
percent_occur_year = util.f2(float(100 * opposite_signs_count) / float(count_same_elements))
print("Percentage of magnetic reconnections:", percent_occur_year + '%')
unique_day_count = util.compare_and_count_opposite_signs2(data_file_1, data_file_2)
average_MR_per_day = round (opposite_signs_count / unique_day_count)
print("Average occurrences of magnetic reconnections per day:", average_MR_per_day)

# processes all relevant data from Wind Spacecraft
print('\n')
print("Wind:")
data_set_name = "Wind"
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_path)
data_file_3 = parent_dir + '/data/wind_5min_bsn_bz_sum.txt'
data_file_4 = parent_dir + '/data/wind_hour_earth_bz.txt'
file_path = parent_dir + '/data/wind_hour_earth_bz.txt'
# used initially to match up files
'''
util.process_file_update_date(data_file_3, data_file_4)
util.add_5th_element(data_file_3, data_file_4)
util.remove_third_elements_inplace(file_path)
'''
count_same_elements = util.find_common_start_and_count(data_file_3, data_file_4)
opposite_signs_count = util.compare_and_count_opposite_signs(data_set_name, data_file_3, data_file_4)
print(f"Number of magnetic reconnections in wind: {opposite_signs_count}") 
percent_occur_year = util.f2(float(100 * opposite_signs_count) / float(count_same_elements))
print("Percentage of magnetic reconnections:", percent_occur_year + '%')
unique_day_count = util.compare_and_count_opposite_signs2(data_file_3, data_file_4)
average_MR_per_day = round (opposite_signs_count / unique_day_count)
print("Average occurrences of magnetic reconnections per day:", average_MR_per_day)

# processes all relevant data from DSCOVR Spacecraft
print('\n')
print("DSCOVR:")
data_set_name = "DSCOVR"
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_path)
data_file_5 = parent_dir + '/data/dscov_5min_bsn_bz_sum.txt'
data_file_6 = parent_dir + '/data/dscov_hour_earth_bz.txt'
file_path = parent_dir + '/data/wind_hour_earth_bz.txt'
# used initially to match up files
'''
util.process_file_update_date(data_file_5, data_file_6)
util.add_5th_element(data_file_5, data_file_6)
util.remove_third_elements_inplace(file_path)
'''
count_same_elements = util.find_common_start_and_count(data_file_5, data_file_6)
opposite_signs_count = util.compare_and_count_opposite_signs(data_set_name, data_file_5, data_file_6)
plt.show()
print(f"Number of magnetic reconnections in dscovr: {opposite_signs_count}") 
percent_occur_year = util.f2(float(100 * opposite_signs_count) / float(count_same_elements))
print("Percentage of magnetic reconnections:", percent_occur_year + '%')
unique_day_count = util.compare_and_count_opposite_signs2(data_file_5, data_file_6)
average_MR_per_day = round (opposite_signs_count / unique_day_count)
print("Average occurrences of magnetic reconnections per day:", average_MR_per_day)

# gse coordinate system conversion to gsm not imolemented, program assumes bz components of gse and gsm are the same
'''
z_gse = 3
x_gse = 1
y_gse = 2

z_gsm = util.gse_z_to_gsm(z_gse, x_gse, y_gse)
print("Z coordinate in GSM:", z_gsm)
'''
