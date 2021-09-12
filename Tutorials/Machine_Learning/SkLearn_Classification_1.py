from math import nan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("/Users/moomacprom1/Data_science/Code/GitHub/Tutorials/Machine_Learning/Data/gssdata.csv").reset_index()
#* Non-null
print(data.shape)
print(data)
print(data.describe())
data_IsNA = data.info() 
print(data_IsNA)

#* Setting up array
neighbors = np.arange(1,35)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
y = data['news'].values
X = data.drop(['news', 'newsreordered', 'id', 'vote96'], axis=1).values

#* Loop Machine Learning
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

    knn.fit(X_train, y_train)

    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
    score = knn.score(X_test, y_test)
    print("Score of Tested: {} for number of {} ".format(score*100, i))
    
#* Visualizations
plt.title('k-NN: Suitable number of K')
plt.plot(neighbors, test_accuracy, label='Testing accuracy')
plt.plot(neighbors, train_accuracy, label='Train accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


"""
Conclusion:  
    The accuracy of model, from testing, is approximately 67.23% for K-value == 14.
This is quite small value. Consider objective of model, this model was constructed in order to see behavior of reader.
By using data from gssdata.csv. To do this objective, accuracy of model have to more than 85%. 
    The reason of small score came from number of row of data is small. This model use only 2765 of row. Moreover, divided into 2 parts,
which are train and test, 0.3 and 0.7, respectively. Therefore, in order to improve accuracy of model, high number of row have to be applied.
"""



