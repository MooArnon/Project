from itertools import count
import numpy as np
from numpy.core.shape_base import stack
from numpy.lib.function_base import quantile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import squarify

#import data
customer_demographic = pd.read_csv(r'/Users/moomacprom1/Data_science/Code/GitHub/Vitual_Intern/KPMG/Data/Clean/Cn_customer_demographic.csv',encoding='windows-1252')
new_customer_list = pd.read_csv(r'/Users/moomacprom1/Data_science/Code/GitHub/Vitual_Intern/KPMG/Data/Clean/Cn_new_customer_list_age.csv')
today_date = dt.date.today().year
#* Age distribution of old and new customers
customer_demographic['DOB'] = pd.to_datetime(customer_demographic['DOB'])
new_customer_list['DOB'] = pd.to_datetime(new_customer_list['DOB'])
age_observing = [new_customer_list, customer_demographic]

data = [new_customer_list, customer_demographic]
# Age distribution
for i in age_observing:
    i['DOB'] = pd.to_datetime(i['DOB'], errors='coerce')
    i['birth_year'] = i['DOB'].dt.year
    i['age'] = today_date - i['birth_year']

def plot_dist(data, x, bin):
    sns.displot(data, x=x, bins=bin)
    plt.xticks(np.arange(0, len(i['age'])+1, 5))
    plt.xlim([0,100])
    plt.show()
plot_dist(customer_demographic, 'age', 70)
plot_dist(new_customer_list, 'age', 30)
# Adding tenure factor
customer_demographic['age*tenure'] = customer_demographic['age']*customer_demographic['tenure']

#* Gender and sales
for i in data:
    gender_count = i['gender'].value_counts()
    total_gender = i['gender'].value_counts().sum()
    ratio_gender = gender_count/total_gender
    #bar
    i['gender'].value_counts().plot(kind="bar", color = (0.3,0.1,0.4,0.6))
    plt.close()
    print(gender_count)
    print(total_gender)
    print(ratio_gender)
    #tree graph
    squarify.plot(sizes=ratio_gender, label=['Female', 'Male', 'Unknown'], alpha=.8 )
    plt.title("Gender Market share'")
    plt.close()


#* Quatile of age vs sales weighted by welth segment
Q1 = customer_demographic[customer_demographic['age']>34]['wealth_segment']
Q2 = customer_demographic[customer_demographic['age']<=34]['wealth_segment']
Q3 = customer_demographic[customer_demographic['age']<=44]['wealth_segment']
Q4 = customer_demographic[customer_demographic['age']>53]['wealth_segment']

age_quatile = [0,0,0,0] # 1stQuatile, 2ndQuatile, 3rdQuatile, 4thQuatile
for i in customer_demographic['age']:
    if i <= 34:
        age_quatile[0] += 1
    elif i <= 44:
        age_quatile[1] += 1
    elif i <= 53:
        age_quatile[2] += 1
    else:
        age_quatile[3] += 1

quatile = []
for i in customer_demographic['age']:
    if i <= 34:
        quatile.append('Q1')
    elif i <= 44:
        quatile.append('Q2')
    elif i <= 53:
        quatile.append('Q3')
    elif i > 53:
        quatile.append('Q4')
    else:
        quatile.append(0)
customer_demographic['quatile'] = quatile

bar_cat_wealth = customer_demographic[['age', 'wealth_segment', 'quatile']]
bar_cat_wealth = bar_cat_wealth[['wealth_segment', 'quatile']].value_counts().reset_index()
bar_cat_wealth = bar_cat_wealth.rename(columns = {0:'count'})
bar_cat_wealth = bar_cat_wealth[bar_cat_wealth.quatile != 0]


sns.catplot(data=bar_cat_wealth, x='quatile', y='count', hue='wealth_segment', kind='bar', palette="Blues_d")
plt.close()

#* job industry
for i in data:
    i = i['job_industry_category'].value_counts()
    i=i.sort_values(ascending=False)
    i.plot.bar()
    plt.close()
    
#* categorize car owned by state
car_owns_count = new_customer_list.pivot_table(index=['state', 'gender'], values='owns_car', aggfunc='count').reset_index()
sns.catplot(data=car_owns_count, x='state', y='owns_car', hue='gender', kind='bar')
sns.color_palette("rocket_r", as_cmap=True)
plt.close()

sns.countplot(data=new_customer_list, x='owns_car')
plt.show()


