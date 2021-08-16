'''
    This project was created this model in order to be used in 'Site eng. helping book'.
        Dear site eng, who want to use this model. You have to install essential package below before use.
        Dear developer, I'm actually new in this branch. I optimized my code to be easy to read.
    However you can contact me anytime by oomarnon.000@kmutt.ac.th. Thank you guys who interested in my project.
'''

from numpy.lib.function_base import append
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.categorical import catplot
import squarify
import matplotlib
from itertools import product

data1 = pd.read_csv(r'/Users/moosmacm1/Data_science/Code/GITHUB/GITHUB/Internship_Project/Visualization/Source/Test.csv')

#* Categorized
#by contractor
data_contractor= data1.pivot_table(index=['Contractors','#Process'], values=['Remaining', 'Quality', 'Factor1', 'Factor2', 'Factor3'])

#* Basic statistic
# List of contractors
contractor = pd.DataFrame(data1['Contractors'].value_counts()).reset_index()
contractors = contractor['index']
# Basic statistic properties
def statistic_contractors(contractor):
    print('====================================================================')
    print('Basic statistic properties of', contractor, 'are')
    print(data1[data1['Contractors'] == contractor].describe())
    print('====================================================================')
# Factor plots
def plot_factor(factor):
    sns.catplot(x='#Process', y=factor, hue='Contractors', kind='bar',data=data1)
    plt.title('%s' %factor)
    plt.show()

#* Tree graph to be used in market share, by score.
# list of contractor names
contractor_name = []
for i in contractors:
    score = []
    contractor_name.append(i)
# group by contractors
score_factors = ['Remaining', 'Quality', 'Factor1', 'Factor2', 'Factor3']
summary_score = data1.pivot_table(index='Contractors', values=score_factors, aggfunc='sum').reset_index()
#adding weighted factor
summary_score['Remaining'] = summary_score['Remaining']*5
#build summary column
summary_score["sum"] = summary_score.sum(axis=1)
total_score = summary_score['sum'].sum()
#build score's market share for each comtractors
summary_score['ratio'] = (summary_score['sum']/total_score)*100
x = summary_score['ratio'].astype('str').str[:5]
label = summary_score['Contractors'] + ' = ' + x + '%'
print(summary_score)
# Plotting tree map graph
def plot_tree():
    squarify.plot(sizes=summary_score['ratio'], label=label, alpha=.8 )
    plt.title("Contractor 'Score's Market share'")
    plt.show()

#* line plot
def plot_remain(contractor):
    data_line_plot = data_contractor.reset_index()
    data_line_plot = data_line_plot[data_line_plot['Contractors']==contractor]
    data_line_plot.plot.line(x='#Process', y='Remaining')
    plt.show()

#* Bell curve
def plot_bell(data):
    def pdf(x):
        mean = np.mean(x)
        std = np.std(x)
        y_out = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2))
        return y_out
    x = data1[data].sort_values()
    y = pdf(x)
    plt.style.use('seaborn')
    plt.figure(figsize = (6, 6))
    plt.plot(x, y, color = 'black',linestyle = 'dashed')
    plt.scatter( x, y, marker = 'o', s = 25, color = 'red')
    plt.title('Bell curve of %s' % data)
    plt.show()

#* Delay bar plot
def remaining_plot(contractor,kind):
    data_delay = data1[['Contractors', 'Remaining', '#Process']]
    data_delay_contractor = data_delay[data_delay['Contractors'] == contractor]
    data_delay_contractor.plot(x='#Process', y='Remaining', kind=kind)
    plt.title('Contractor %s' % contractor)
    plt.xticks(rotation=45)
    plt.show()


