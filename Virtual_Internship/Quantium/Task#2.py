import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew #for some statistics
import re
from IPython.display import display
import warnings
import matplotlib.dates as mdates
import sklearn

warnings.simplefilter(action='ignore', category=FutureWarning)

#* Import data
final_data = pd.read_csv("/Users/moomacprom1/Data_science/Code/MacProM1/Virtual_Intern/Quantium/Data/final_data.csv", parse_dates= ['DATE'])

#* Rearrange date in month and years 
final_data.insert(1, 'year_month',final_data['DATE'].dt.to_period('M'))     # .dt.to_period('M')

#* Change type of artribute
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
del check
##drop
indices_to_drop = data[data['STORE_NBR'].isin(stores_with_less_than_12_months)].index
data = data.drop(indices_to_drop)

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
del check, colors

#* Group data by metric
data['yearly_sale'] = data.groupby('STORE_NBR')['TOT_SALES'].transform('sum')
data['yearly_cust'] = data.groupby('STORE_NBR')['LYLTY_CARD_NBR'].transform('nunique')
data['monthly_sale'] = data.groupby(['STORE_NBR', 'year_month'])['TOT_SALES'].transform('sum')
data['monthly_cust'] = data.groupby(['STORE_NBR', 'year_month'])['LYLTY_CARD_NBR'].transform('nunique')

avg_trans = data.groupby('STORE_NBR').apply(lambda subdf: (subdf['TXN_ID'].nunique() / subdf['yearly_cust'].unique()))
avg_trans = avg_trans.astype('float64')
data['avg_txn_per_cust'] = data['STORE_NBR'].map(avg_trans)

pre_df = data[data['DATE'] < "2019-02-01"]
trial_df = data[(data['DATE'] > "2019-01-31") & (data['DATE'] < "2019-05-01")]

Trial_MinDate, Trial_MaxDate = min(trial_df['DATE']), max(trial_df['DATE'])
PreTrial_MinDate, PreTrial_MaxDate =  min(pre_df['DATE']), max(pre_df['DATE'])
print(f'the trial_df dataframe consists of samples between {Trial_MinDate}, {Trial_MaxDate}')
print(f'the pre_df dataframe consists of samples between {PreTrial_MinDate}, {PreTrial_MaxDate}')

corrmat = pre_df.corr()
mask = np.triu(np.ones_like(corrmat, dtype=bool))
# plt.subplots(figsize = (25, 15))
sns.heatmap(corrmat, mask = mask, cmap = 'coolwarm', annot = True)
plt.xticks(rotation = 45)
plt.show()

metrics_cols = ['STORE_NBR', 'year_month', 'yearly_sale',
                'yearly_cust','monthly_sale', 'monthly_cust', 'avg_txn_per_cust']


def extract_metrics(df):
    subdf = df.loc[:, metrics_cols].set_index(['STORE_NBR', 'year_month']).sort_values(by = ['STORE_NBR', 'year_month'])
    subdf.drop_duplicates(inplace = True, keep = 'first')
    return subdf 

metrics_df = extract_metrics(pre_df)
print(metrics_df.head())

def calc_corr(trial_store):
    a=[]
    metrics = metrics_df[['monthly_sale', 'monthly_cust']] 
    for i in metrics.index:
        a.append(metrics.loc[trial_store].corrwith(metrics.loc[i[0]]))
    subdf = pd.DataFrame(a)
    subdf.index = metrics.index
    subdf = subdf.drop_duplicates()
    subdf.index = [s[0] for s in subdf.index]
    subdf.index.name ="store_nbr"
    subdf = subdf.abs()
    subdf['mean_corr'] = subdf.mean(axis=1)
    subdf.sort_values(by = 'mean_corr', ascending = False, inplace = True)
    return subdf

##*Finish all metric model build

#*
'''
print('Corelation Score of store 77')
print(calc_corr(77).drop(77).head(5))

print('Corelation Score of store 86')
print(calc_corr(86).drop(86).head(5))

print('Corelation Score of store 88')
print(calc_corr(88).drop(88).head(5))

    Control store   |   Trial store
        77          |       233
        86          |       155
        88          |       178
'''

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
    plt.show()
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

