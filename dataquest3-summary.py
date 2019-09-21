#%% [markdown]
# # Pandas and Numpy Fundamentals
# ## Introduction to NumPy
# ### Selecting Rows, Columns, and Items from an Ndarray

#%%
# Convert a list of lists into a ndarray
import numpy as np
import csv
f = open("/Users/kevinmtaing/Documents/python_vscode/dataquest3-summary/nyc_taxis.csv", "r")
taxi_list = list((csv.reader(f)))
taxi_list = taxi_list[1:] # remove the header row

# convert all values to floats
converted_taxi_list = []
for row in taxi_list:
    converted_row = []
    for item in row:
        converted_row.append(float(item))
    converted_taxi_list.append(converted_row)
taxi = np.array(converted_taxi_list)

#%%
# Number of row and column in list (numpy.ndarray)
taxi_shape = np.shape(taxi)
taxi_shape = taxi.shape
print(taxi_shape)

#%%
# Selecting the data
select1 = taxi[0] # first row
select2 = taxi[391:501] # row 391 to 500
select3 = taxi[21,5] # row 21 column 5
select4 = taxi[:,[1,4,7]] # column 1, 4, 7
select5 = taxi[99,5:9] # row 99 column 5 to 8
select6 = taxi[100:201, 14] # row 100 to 200 column 14
select7 = taxi[:,4] # column 4

cols = [1,3,5]
select8 = taxi[:, cols]

#%% [markdown]
# Vector Math
# 
# `vector_a + vector_b`: Addition
#
# `vector_a - vector_b`: Subtraction
#
# `vector_a * vector_b`: Multiple
# 
# `vector_a / vector_b`: Division

#%% [markdown]
# Calculating Statistics for 1D Ndarrays
#
# `ndarray.min()` to calculate the minimun value
# 
# `ndarray.max()` to calculate the maximun value
# 
# `ndarray.mean()` to calculate the mean average value
# 
# `ndarray.sum()` to calcualte the sum of the values


#%%
# Calculating Statistics for 2D Ndarrays
taxi.max() # max value for an entire 2D Ndarray
taxi.max(axis = 1) # max value for each row in a 2D Ndarray (return a 1D Ndarray)
taxi.max(axis = 0) # max value for each column in a 2D Ndarray (return a 1D Ndarray)

#%%
