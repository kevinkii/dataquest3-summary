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

#%% [markdown]
# ## Boolean Indexing with Numpy
# ### Reading CSV Files with NumPy

#%%
# Reaing in a CSV file
import numpy as np
taxi = np.genfromtxt("/Users/kevinmtaing/Documents/python_vscode/dataquest3-summary/nyc_taxis.csv", delimiter=",", skip_header=1)

# ### Boolean Arrays

#%%
# Creating a Boolean array from filtering criteria
np.array([2,4,6,8]) < 5

#%%
# Boolean filtering for 1D ndarray
a = np.array([2,4,6,8])
filter = a < 5
a[filter] # return only "true"

#%%
# Boolean filtering for 2D ndarray
tip_amount = taxi[:,12]
tip_bool = tip_amount > 50
top_tips = taxi[tip_bool, 5:14]

#%% [markdown]
# ### Assigning Values

#%%
# Assigning values in a 2D ndarray using indices
taxi[28214,5] = 1
taxi[:,0] = 16
taxi[1800:1802,7] = taxi[:,7].mean()

#%%
# Assigning values using Boolean arrays
new_column = np.zeros([taxi.shape[0],1]) # createa a new column with `0`
taxi = np.concatenate([taxi,new_column], axis=1) # add `new_column` to `taxi`
taxi[taxi[:,5] == 2, 15] = 1

#%%
pickup_month = taxi[:,1]
january_pool = pickup_month == 1
january = pickup_month[january_pool]
january_rides = np.shape(january)[0] # return as integer

#%%
# Assigning values using Boolean arrays
total_amount = taxi[:,13]
total_amount[total_amount<0] = 0

#%%
# Copy ndarray
taxi_copy = taxi.copy()

#%% [markdown]
# ## Introduction to Pandas
# ### Pandas Dataframe Basics
#
# | **Select by Label** | **Explicit Syntax** | **Shorthand Convention** |
# | --- | --- | --- |
# | Single Column from dataframe | `df.loc[:,"col1"]` | `df["col1"]` |
# | List of columns from dataframe | `df.loc[:,["col1","col7"]]` | `df[["col1","col7"]]` | 
# | Slice of coulmns from datafram | `df.loc[:,"col1":"col4"` | |
# | Single row from dataframe | `df.loc["col4"]` | |
# | List of rows from dataframe | `df.loc[["row1","row8"]]` | |
# | Slice of rows from dataframe | `df.loc["row3":"row5]"` | `df["row3":"row5"]` |
# | Single item from series | `s.loc["item8"]` | `s["item8"]` |
# | List of items from series | `s.loc[["item1","item7"]]` | `s[["item1","item7"]]` |
# | Slice of items from series | `s.loc["item2":"item4"`] | `s["item2":"item4"]` |

#%% [markdown]
# Series Math
# 
# `series_a + series_b`: Addition
#
# `series_a - series_b`: Subtraction
#
# `series_a * series_b`: Multiple
# 
# `series_a / series_b`: Division

#%% [markdown]
# Calculating Statistics for Series and Dataframe
#
# `series.min()` and `dataframe.max()`
# 
# `series.max()` and `dataframe.min()`
# 
# `series.mean()` and `dataframe.mean()`
# 
# `series.median()` and `dataframe.median()`
#
# `series.mode()` and `dataframe.mode()`
#
# `series.sum()` and `dataframe.sum()`

#%% [markdown]
# Calculating the method by row or column
#
# `dataframe.method(axis="index")` # by row
#
# `dataframe.method(axis="column")` # by column

#%%
# Reading a file into a dataframe
import pandas as pd
import numpy as np
dtaxi = pd.read_csv("/Users/kevinmtaing/Documents/python_vscode/dataquest3-summary/nyc_taxis.csv", index_col=0)
dtaxi

#%%
# Returning a dataframe's data type
col_types = dtaxi.dtypes
col_types

#%%
# Returning the dimensions of a dataframe
dims = dtaxi.shape
dims

#%% [markdown]
# ### Selecting Values from a Dataframe

#%%
# Selecting a single column
dtaxi["pickup_month"]

#%%
# Selecting multiple columns
dtaxi[["pickup_year","pickup_month"]]

#%%
dtaxi.head(5) # select first 5 rows
dtaxi.tail(5) # select last 5 rows

#%% [markdown]
# ## Exploring Data with Pandas: Fundamental
# ### Data Exploration Methods

#%%
# Describing a series object
f500 = pd.read_csv("/Users/kevinmtaing/Documents/python_vscode/dataquest3-summary/f500.csv", index_col=0)
revs = f500["revenues"]
summary_revs = revs.describe()
summary_revs

#%%
# Unique value counts for a column
country_freq = f500['country'].value_counts()
country_freq_chn = f500['country'].value_counts().loc['China']


#%% [markdown]
# ### Assignment with Pandas

#%%
# Creating a new column
f500["year_founded"] = 0

#%%
# Replacing a specific value in a dataframe
f500.loc["Dow Chemical","ceo"] = "Jim Fitterling"

#%% [markdown]
# ### Boolean Indexing in Pandas

#%%
# Filtering a dataframe down on a specific value in a column
kr_bool = f500["country"] == "South Korea"
top_5_kr = f500[kr_bool].head()

#%%
# Updating values using Boolean Filtering
f500.loc[f500["previous_rank"] == 0, "previous_rank"] = np.nan
prev_rank_after = f500["previous_rank"].value_counts(dropna=False).head()

#%%
rank_change = f500["previous_rank"] - f500["rank"]
rank_change_max = rank_change.max()
rank_change_min = rank_change.min()

#%%
top5_rank_revenue = f500[["rank","revenues"]].head()
print(top5_rank_revenue)

#%%
