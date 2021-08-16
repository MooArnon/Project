import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

customer_address = pd.read_csv(r'/Users/moomacprom1/Data_science/Code/GitHub/Vitual_Intern/KPMG/Data/Raw/C_Customer_adress.csv', encoding='windows-1252')
customer_demographic = pd.read_csv(r'/Users/moomacprom1/Data_science/Code/GitHub/Vitual_Intern/KPMG/Data/Raw/C_Customer_demographic.csv', encoding='windows-1252')
new_customer_list = pd.read_csv(r'/Users/moomacprom1/Data_science/Code/GitHub/Vitual_Intern/KPMG/Data/Raw/C_New_customer_list.csv', encoding='windows-1252')
transaction = pd.read_csv(r'/Users/moomacprom1/Data_science/Code/GitHub/Vitual_Intern/KPMG/Data/Raw/C_Transaction.csv', encoding='windows-1252')

data_list = [customer_demographic['DOB'], new_customer_list['DOB'], transaction['transaction_date']]
#* Check outing date
customer_demographic['DOB'] = pd.to_datetime(customer_demographic['DOB'])
new_customer_list['DOB'] = pd.to_datetime(new_customer_list['DOB'])
transaction['transaction_date'] = pd.to_datetime(transaction['transaction_date'])
today_date = dt.date.today()
# Check that birthday and transaction date is not be at future
assert customer_demographic['DOB'].max().date() <= today_date
assert new_customer_list['DOB'].max().date() <= today_date
assert transaction['transaction_date'].max().date() <= today_date

today_date = dt.date.today().year
#* Adding age column
age_observing = [new_customer_list, customer_demographic]
for i in age_observing:
    i['DOB'] = pd.to_datetime(i['DOB'], errors='coerce')
    i['birth_year'] = i['DOB'].dt.year
    i['age'] = today_date - i['birth_year']

data_list = [customer_address, customer_demographic, new_customer_list, transaction]
def basic_properties(data):
    print('head')
    for i in data:
        print('================== head of ======================================================')
        print(i.head())
        print('==================== end =====================================================')
    print('shape')
    for i in data:
        print(i.shape)


#** Customer address
customer_address_column_list = ['state', 'country', 'property_valuation']
def customer_address_count_plot(data):
    for i in data:
        sns.countplot(data=customer_address, x=i)
        plt.title(i)
        plt.show()
'''
OK for country and property values
For state, there have some same data but use differences key words, which Victoria is VIC and New South Wales is NSW. These mistake have to be changed
'''
customer_address['state'] = customer_address['state'].replace({'New South Wales':'NSW', 'Victoria':'VIC'})      #OK


#** Customer demographic
#drop default column
customer_demographic = customer_demographic.drop(columns='default')
#fill missing values with NaN
customer_demographic.fillna(method='ffill')
#use count plot to find outlirers
demo_interested_col_list = ['gender', 'job_industry_category', 'wealth_segment', 'deceased_indicator', 'owns_car', 'tenure']
def customer_demo_count_plot(data):
    for i in data:
        sns.countplot(data=customer_demographic, x=i)
        plt.xticks(rotation=45)
        plt.show() 
#Gender: Femal and F is Female while M is male
customer_demographic['gender'] = customer_demographic['gender'].replace({'Femal':'Female', 'F':'Female', 'M':'Male'}) #OK


#** New customer lists
#fill missing value
new_customer_list = new_customer_list.fillna(method='ffill')
#use count plot to find outlirers
new_customer_col_list = ['gender', 'past_3_years_bike_related_purchases', 'job_industry_category', 
                        'wealth_segment', 'owns_cars', 'state', 'country', 'property_valuation']
def new_customer_count_plot(data):
    for i in data:
        sns.countplot(data=new_customer_list, x=i)
        plt.xticks(rotation=90)
        plt.show()
#No mistake in this section

#** Transaction
transaction = transaction.fillna(method='ffill')
##use count plot to find outlirers
transaction_col_list = ['online_order', 'order_status', 'brand', 'product_line', 'product_class', 'product_size']
def transaction_count_plot(data):
    for i in data:
        sns.countplot(data=transaction, x=i)
        plt.xticks(rotation=90)
        plt.show()
#No mistake in this section


#exprot data
'''
customer_address.to_csv('Cn_customer_address.csv')
customer_demographic.to_csv('Cn_customer_demographic.csv')
new_customer_list.to_csv('Cn_new_customer_list.csv')
'''
print(customer_demographic['gender'])

print('No error')