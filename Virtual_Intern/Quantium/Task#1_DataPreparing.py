import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import re as re

purchase_behavior = pd.read_csv(r'MacProM1/Virtual_Intern/Quantium/Data/QVI_purchase_behaviour.csv')
transaction = pd.read_csv(r'/Users/moomacprom1/Data_science/Code/MacProM1/Virtual_Intern/Quantium/Data/QVI_transaction_data.csv')

data_list = [purchase_behavior, transaction]

#* Transaction
# Date 
today_date = dt.date.today()
transaction['DATE'] = pd.TimedeltaIndex(transaction['DATE'], unit='d')+ dt.datetime(1899,12,30)
transaction['DATE'] = pd.to_datetime(transaction['DATE']).dt.normalize()
transaction = transaction.sort_values(by='DATE')
assert transaction['DATE'].dtype == 'datetime64[ns]'
assert transaction.DATE.max() <= today_date
## Missing date ?
date_check= transaction.set_index('DATE')
date_check.index= pd.to_datetime(date_check.index)
print(pd.date_range(start='2018-07-02', end='2019-06-30').difference(date_check.index)) # To see missing date.
### 2018-12-25 is missing. In order to check the typo visualization have to be used
sns.countplot(data=transaction, x='DATE')
plt.xticks(rotation=90)
plt.close()
"""
    From date cleaning. There are not containing future date. However one date is missing, which is 2018-12-25, Christmass day.
    Then, after visualing number of transaction per day. Behavior of customer buying show the increment of sales be for Christmass.
    So, this missing can be ignored
"""

# Check PRO_QTY and TOT_SALES
QTY_SALES_outlirers = ['PROD_QTY', 'TOT_SALES']
for i in QTY_SALES_outlirers:
    plt.scatter(x=transaction['DATE'], y=transaction[i], )
    plt.close()
"""
    They are two outlirers for these two sections. Which two values got over 200 units of PROD_QTY, related to TOT_SALES, same days.
    This may be charity purpose or marketing staff. Which can be dropped.
"""
transaction = transaction[transaction['PROD_QTY'] < 190]
for i in transaction['PROD_QTY']:
    assert i < 190
print('No more outlirers.')

# Extract brand name and bag size from PROD_NAME
bag_brand= pd.DataFrame(transaction.PROD_NAME.str.split().str.get(0))
transaction['Brand_name'] = bag_brand
transaction['Brand_name'] = bag_brand
sns.countplot(data=transaction, x='Brand_name') # Visualize brand name
plt.xticks(rotation=90)
plt.close
## Some of brand name look like they are same brands. In order to precise data, this typo must be fixed.
brand_dictionary={'RRD':'Red','Infzns':'Infuzions', 'Smith':'Smiths', 'WW':'Woolworth', 
                'Snbts':'Sunbites','Dorito':'Doritos','GrnWves':'Grain' }
transaction_datai=bag_brand.replace(brand_dictionary)
bag_brandi= transaction_datai.PROD_NAME.str.split().str.get(0)
transaction['Brand_name']=bag_brandi
for i in transaction['Brand_name']:
    assert i != ['Red', 'Infuzions', 'Smiths', 'Woolworth', 'Sunbites', 'Doritos', 'Grain']
### Bag sizes
def find_number(text):
    num= re.findall(r'[0-9]+',text)
    return " ".join(num)
transaction["Bag_size"] = transaction['PROD_NAME'].apply(lambda x: find_number(x))
#### OK for PROD_NAME section
##* OK For Transaction data


#* Purchase Behavior
observe_PurchaseBehavior_col = ['LIFESTAGE', 'PREMIUM_CUSTOMER']
for i in observe_PurchaseBehavior_col:
    sns.countplot(data=purchase_behavior, x=i)
    plt.close()
##* OK For purchase_behavior

#* Merge two data
final_data= transaction.merge(purchase_behavior, on="LYLTY_CARD_NBR")
print(final_data.head())
print(final_data.shape)
print(transaction.shape)
##* No duplication

#* Export data
final_data.to_csv('final_data.csv')
print('################################## No Error ####################################')