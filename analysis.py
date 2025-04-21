'''
File to process and analyze the data from the Blind Dating event.

The main goal is to determine which questions where useful in predicting a good date
based on a number of different outcomes.

Although "dating" might be the ideal outcome, seeing if people found compatability or
a new friend through the process should also be considered a success as the questions
largely asked about personality compatability.

Any findings should be considered in determining which questions to change, add, and remove.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import the csv files response.csv and postResponse.csv
response = pd.read_csv('response.csv')
postResponse = pd.read_csv('postResponse.csv')

# print the first 5 rows of each dataframe
print(response.head())