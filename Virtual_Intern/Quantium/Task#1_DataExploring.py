from typing import final
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import squarify

final_data = pd.read_csv(r'/Users/moomacprom1/Data_science/Code/MacProM1/Virtual_Intern/Quantium/Data/final_data.csv')

#* Bag sizes and Brand counting
# Size
sns.countplot(data=final_data, x='Bag_size' )
plt.title('Bag sizes sales distribution')
plt.xticks(rotation = 45)
plt.close()
# Brand name
sns.countplot(data=final_data, x='Brand_name')
plt.title('Brand sales distribution')
plt.xticks(rotation = 45)
plt.close()
# size of top brand
top_brand = ['Kettle', 'Smiths', 'Doritos']
for i in top_brand:
    top_brand_size = final_data.set_index('Brand_name')
    top_brand_size = top_brand_size.loc[i].reset_index()
    sns.countplot(data=top_brand_size, x='Bag_size')
    plt.close()
"""
    The most popular bag sizes for surveying are 175g, 150g, 134g and 110g respectively. So, approximately 150g of bag sizes are popular.
    Others section is over 300g which are 300g, 330g and 380g also have high sales.
    Kettle, Smiths and Doritos is top 3 highest sales for this transaction data
    After merging data, popular brand contained popular bag size
"""

#* PROD_QTY
sns.countplot(data=final_data, x='PROD_QTY')
plt.title('Count number of sales quantitis per one transaction')
plt.close()
"""
    popular quantity sales per transaction are 1 and 2 pack, with extremely value.
"""


#* LIFESTAGE and PREMIUM_CUSTOMER
# Create a data frame with fake data
LifeStage_PremiumCustomer_Count = final_data[['LIFESTAGE', 'PREMIUM_CUSTOMER']].value_counts().reset_index()
LifeStage_PremiumCustomer_Count['LifestagePremiumcustomer'] = LifeStage_PremiumCustomer_Count['LIFESTAGE'] + ' +\n ' +LifeStage_PremiumCustomer_Count['PREMIUM_CUSTOMER']
print(LifeStage_PremiumCustomer_Count)

#build summary column
LifeStage_PremiumCustomer_Count["sum"] = LifeStage_PremiumCustomer_Count.sum(axis=1)
total_score = LifeStage_PremiumCustomer_Count['sum'].sum()
#build score's market share for each comtractors
LifeStage_PremiumCustomer_Count['ratio'] = (LifeStage_PremiumCustomer_Count['sum']/total_score)*100
x = LifeStage_PremiumCustomer_Count['ratio'].astype('str').str[:5]
label = LifeStage_PremiumCustomer_Count['LifestagePremiumcustomer'] + ' = ' + x + '%'
print(LifeStage_PremiumCustomer_Count)
# Plotting tree map graph
squarify.plot(sizes=LifeStage_PremiumCustomer_Count['ratio'], label=label, alpha=.8 )
plt.rcParams.update({'font.size': 15})
plt.title("Market share grouped by Lifestage and Customer types")
plt.close()
"""
Top 3
1. Older families + Budget = 8.745%
2. Retireses + Mainstream = 8.105%
3. Young singles/couples + Mainstream = 7.874%
"""

#* Popular Brand for TOP3
# Budget older families
def top3_brand(LifeStage, PremiumCustomer):
    top3 = final_data[final_data['LIFESTAGE'] == LifeStage]
    top3 = final_data[final_data['PREMIUM_CUSTOMER'] == PremiumCustomer]
    sns.countplot(data=top3, x='Brand_name')
    plt.xticks(rotation = 45)
    plt.show()


"""
    Conclusion: Top 3 valuable customers are Budget older families, Mainstream retireses and Mainstream young singles/couples.
                Customers almost buy 2 and 1 packs per transaction. Moreover, Kattle is the most popular brand with approximately 150 grams
                of package. 
"""