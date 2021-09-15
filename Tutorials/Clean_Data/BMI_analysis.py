import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.indexing import check_bool_indexer
import seaborn as sns
from fuzzywuzzy import process

#* import data
data = pd.read_csv("/Users/moomacprom1/Data_science/Code/GitHub/Tutorials/Clean_Data/Data/BMI_Surveying.csv")
data = data.drop(['Unnamed: 1', 'Unnamed: 2'], axis=1)
print(data.shape)
print(data.head())
print(data.describe())
print(data.columns)
##* finish importing process

#* Delete all thai language
# Gender
categories_gender = ['Male', 'Female']
for gender in categories_gender:
	matches = process.extract(gender, data['Gender'], limit=data.shape[0])
	for potential_match in matches:
		if potential_match[1] >= 80:
			data.loc[data['Gender'] == potential_match[0], 'Gender'] = gender

print(data["Gender"])

	


