"""
    Objective: Measure the efficiency of trial store compare with control store
    Data: From Quantium data analytics visual experiences program
    By: Arnon Phongsiang
    Notation: I use this set of problems to study workflow and how data scientist work.
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew #for some statistics
import re
from IPython.display import display
import matplotlib.dates as mdates

#* Import data
final_data = pd.read_csv("/Users/moomacprom1/Data_science/Code/GitHub/Virtual_Internship/Quantium/Data/clean/final_data.csv", parse_dates= ['DATE'])
print(final_data.head(5))
""" The final_data is
       Unnamed: 0       DATE  STORE_NBR  LYLTY_CARD_NBR  TXN_ID  ...  TOT_SALES Brand_name  Bag_size              LIFESTAGE PREMIUM_CUSTOMER
0           0 2018-07-01         19           19205       16466  ...        3.7   Pringles       134  OLDER SINGLES/COUPLES       Mainstream
1           1 2018-08-19         19           19205       16467  ...        8.4     Kettle       135  OLDER SINGLES/COUPLES       Mainstream
2           2 2018-09-18         19           19205       16468  ...       10.8     Kettle       175  OLDER SINGLES/COUPLES       Mainstream
3           3 2018-07-01        189          189381      190189  ...        3.1      Grain       180         OLDER FAMILIES       Mainstream
4           4 2018-07-01        124          124236      127984  ...        3.8  Infuzions       110         OLDER FAMILIES           Budget


    The idea is group data by metric, monthly sales and monthly customers. Divide month of ordinary store before and after trial suggestion.
    In order to find control store of each trial store, using each metric in order to measure behavior of each store. 
    Ploting graph to see and analyze behavior.
"""

#* Rearrange date in month and years 
# insert month column 
final_data.insert(1, 'year_month',final_data['DATE'].dt.to_period('M'))   # .dt.to_period('M')

#* Change type of artribute
# Change str features to be category variables
cols_with_changed = {'PROD_NAME': 'category', 'PROD_QTY': 'category',
                           'LIFESTAGE': 'category', 'PREMIUM_CUSTOMER': 'category',
                           'Brand_name': 'category'}
data = final_data.astype(cols_with_changed)
print(data.info())

#* Drop data, which less than 12 months
#check
check = data.groupby('STORE_NBR')['year_month'].nunique()
check = check[check != 12]
print('Stores with less than 12 month transaction data:')
display(check)
stores_with_less_than_12_months = check.index.to_list()
del check # Clear unused check
## drop less than 12 month stores  
indices_to_drop = data[data['STORE_NBR'].isin(stores_with_less_than_12_months)].index
data = data.drop(indices_to_drop) # Drop by index

#* Compare total sales of each trial store
check = data[data['STORE_NBR'].isin([77, 86, 88])].groupby(['STORE_NBR', 'year_month'])['TOT_SALES'].sum()
colors = []
for store, month in check.index:
    if store == 77:
        colors.append('b')
    elif store == 86:
        colors.append('g')
    else:
        colors.append('r')
check.plot(kind = 'bar', color = colors, figsize = (20, 9))
plt.close()
# 77 -> 86 -> 88, lowest to highest
del check, colors

#* Group data by metric
# Adding metric column on each store
data['yearly_sale'] = data.groupby('STORE_NBR')['TOT_SALES'].transform('sum')
data['yearly_cust'] = data.groupby('STORE_NBR')['LYLTY_CARD_NBR'].transform('nunique')
data['monthly_sale'] = data.groupby(['STORE_NBR', 'year_month'])['TOT_SALES'].transform('sum')
data['monthly_cust'] = data.groupby(['STORE_NBR', 'year_month'])['LYLTY_CARD_NBR'].transform('nunique')
print(data)

# Find everage transaction per customer of each store
avg_trans = data.groupby('STORE_NBR').apply(lambda subdf: (subdf['TXN_ID'].nunique() / subdf['yearly_cust'].unique()))
avg_trans = avg_trans.astype('float64')
data['avg_txn_per_cust'] = data['STORE_NBR'].map(avg_trans)
print(data)

# Separate data before and after trial phase
pre_df = data[data['DATE'] < "2019-02-01"] # before
trial_df = data[(data['DATE'] > "2019-01-31") & (data['DATE'] < "2019-05-01")] # after
Trial_MinDate, Trial_MaxDate = min(trial_df['DATE']), max(trial_df['DATE'])
PreTrial_MinDate, PreTrial_MaxDate =  min(pre_df['DATE']), max(pre_df['DATE'])
print(f'the trial_df dataframe is samples between {Trial_MinDate}, {Trial_MaxDate}')
print(f'the pre_df dataframe is samples between {PreTrial_MinDate}, {PreTrial_MaxDate}')

# Finding correlation between each features by using heatmap.
corr_df = pre_df.corr() # Construct the correlation dataframe
mask = np.triu(np.ones_like(corr_df, dtype=bool)) # This line came from stack overflow. It can pass this argument, but it will return normal heatmap, which is so eyestrain.
sns.heatmap(corr_df, mask = mask, cmap = 'YlGnBu', annot = True) # Build heatmap graph
plt.xticks(rotation = 45)
plt.show()

# Determine the metric features
metrics_cols = ['STORE_NBR', 'year_month', 'yearly_sale',
                'yearly_cust','monthly_sale', 'monthly_cust', 'avg_txn_per_cust']

# Fn for construct the useable data frame.
def extract_metrics(df): 
    sub_df = df.loc[:, metrics_cols].set_index(['STORE_NBR', 'year_month']).sort_values(by = ['STORE_NBR', 'year_month'])
    sub_df.drop_duplicates(inplace = True, keep = 'first')
    return sub_df 

metrics_df = extract_metrics(pre_df)
print(metrics_df.head())

# Fn that calculate score of each store, compare with trial store.
def calc_corr(trial_store):
    a=[]
    metrics = metrics_df[['monthly_sale', 'monthly_cust']] 
    for i in metrics.index:
        a.append(metrics.loc[trial_store].corrwith(metrics.loc[i[0]]))
    sub_df = pd.DataFrame(a)
    sub_df.index = metrics.index
    sub_df = sub_df.drop_duplicates()
    sub_df.index = [s[0] for s in sub_df.index]
    sub_df.index.name ="store_nbr"
    sub_df = sub_df.abs()
    sub_df['mean_corr'] = sub_df.mean(axis=1)
    sub_df.sort_values(by = 'mean_corr', ascending = False, inplace = True)
    return sub_df
##*Finish all metric model build

#* Calculate the control store of each trial store
print('Corelation Score of store 77')
print(calc_corr(77).drop(77).head(5))

print('Corelation Score of store 86')
print(calc_corr(86).drop(86).head(5))

print('Corelation Score of store 88')
print(calc_corr(88).drop(88).head(5))

"""
    Control store   |   Trial store
        77          |       233
        86          |       155
        88          |       178
"""

#* See behavior of control and trial stores
# Set trial and control store pandas
final_data = final_data.set_index('STORE_NBR')
data_77_233 = final_data.loc[[77,233]].reset_index()
data_86_155 = final_data.loc[[86,155]].reset_index()
data_88_178 = final_data.loc[[88,178]].reset_index()
data_trial_control = [data_77_233, data_86_155, data_88_178]
# Number of customers
for i in data_trial_control:
    data_plot = i.pivot_table(index=['year_month', 'STORE_NBR'], values='TXN_ID', aggfunc='count').reset_index()
    print(data_plot)
    sns.catplot(data=data_plot, x='year_month', hue='STORE_NBR', y='TXN_ID', kind='point' )
    plt.close()

"""
    From data visualization, in number of customer section. Trial stores haves specific number of customer per month, in trial month, Feb-2019 to Apl-2019, 
    more than control store.The number of customer alw  ays decrease on January of each store, as a usual. 
    However, on control month, the increment of customer is specific in term of applied suggestion from Task#1.
"""

## Total sales
for i in data_trial_control:
    data_plot = i.pivot_table(index=['year_month', 'STORE_NBR'], values='TOT_SALES', aggfunc='sum').reset_index()
    print(data_plot)
    sns.catplot(data=data_plot, x='year_month', hue='STORE_NBR', y='TOT_SALES', kind='point' )
    plt.close()
    
"""
    This section, Total sales, also give us the same result as Number of customer section. All trial stores gain more sales.
"""

"""
    Conclusion: From determining control stores for each trial stores, by giving score in order to find the nearest data values.
                We have 233, 155, 178 of control stores for 77, 86, 88 of trial stores.
                    Control store   |   Trial store
                        77          |       233
                        86          |       155
                        88          |       178
                Then, data visualization will be used in order to measure efficiency of Task#1's policy. 
                After using two metrics, number of customer and total sales.Every metrics of trial stores got higher efficiency compare with control stores.
"""

