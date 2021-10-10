from re import VERBOSE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.api._v2.keras import callbacks
from tensorflow.python.keras.backend import print_tensor
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils.layer_utils import print_summary

#* Import data
data = pd.read_csv('/Users/moomacprom1/Data_science/Code/GitHub/Tutorials/Machine_Learning/Reading_Behavior/Data/gssdata.csv')
# check basic properties
print(data.shape)
print(data.head(5))
print(data.columns)
print(data.isna().sum())
"""
    No null value in this dataset. Moreover, some of features is not have specific relationship with the data. Consequently, we can drop its.
    All data is in Keys, its mean constructor was use it to perform SPSS statistical analysis.
"""

#* Data preprocessing
# Drop unnecessary features
data_drop = data.drop(['vote96', 'vote00', 'usewww', 'zodiac', 'degree3', 'newsreordered', 'id', 'year'], axis=1)
print(data_drop.columns) # OK
# Split train and test dataset
train_data = data_drop.sample(frac=0.8, random_state=0)
test_data = data_drop.drop(train_data.index)
# Sepperate label
train_label = train_data['news']
test_label = test_data['news']
print(test_label)
# Check statistical properties
train_stats = train_data.describe()
train_stats = train_stats.transpose()
print(train_stats)
# Normalize data
def norm(x):
    return (x - train_stats['mean']/train_stats['std'])
normed_train_data = norm(train_data)
normed_test_data = norm(test_data)
print(normed_test_data)

#* Model

model = keras.Sequential([
        tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_data.keys())]), # input shape is number of features in dataset
        tf.keras.layers.Dense(64, activation=tf.nn.relu,),
        tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.RMSprop(0.001) # Learning rate == 0.001

model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
print(model.summary())

Epochs = 1000

model.fit(normed_train_data, train_label, epochs=Epochs)

"""
    Epoch 1000/1000
    70/70 [==============================] - 0s 313us/step - loss: 4.5466e-04 - mae: 0.0095 - mse: 4.5466e-04
"""

