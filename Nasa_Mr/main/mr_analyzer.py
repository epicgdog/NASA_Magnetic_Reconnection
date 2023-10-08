'''
Created on Oct 7, 2023

@author: wliu
'''

import util.functions as util
import numpy as np
import os, sys

curr_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_path)
data_file_1 = parent_dir +  '/data/dscov_5min_bsn_bz.txt'
data_file_2 = parent_dir +  '/data/dscov_5min_bsn_bz_average.txt'


#util.process_file(data_file_1, data_file_2)
# Example usage
util.add_5th_element(data_file_1, data_file_2)


z_gse = 3
x_gse = 1
y_gse = 2

z_gsm = util.gse_z_to_gsm(z_gse, x_gse, y_gse)
print("Z coordinate in GSM:", z_gsm)

