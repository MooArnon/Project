from re import VERBOSE, X
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow._api.v2 import nn
from tensorflow.keras import layers
from tensorflow.python.keras.api._v2.keras import callbacks
from tensorflow.python.keras.backend import print_tensor, relu
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers.core import Dense
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
# Build model
def build_model():
    model = keras.Sequential([
            tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=[len(train_data.keys())]), # input shape is number of features in dataset
            tf.keras.layers.Dense(16, activation=tf.nn.relu),
            tf.keras.layers.Dense(8, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)])
    # Determine optimizer
    optimizer = tf.keras.optimizers.RMSprop(0.001) # Learning rate == 0.001
    # Compile model
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    print(model.summary())
    return model
# stop when model fitted
early_stop = keras.callbacks.EarlyStopping(monitor='mae', patience=200)
# fitting model
model = build_model()
# store history of trained
history = model.fit(normed_train_data, train_label, epochs=1000, validation_split=0.2, callbacks=early_stop)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch


#* Visualize how fitted of model
def model_plotted(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.xlabel('epoch')
    plt.ylabel('Mean App Err.')
    plt.plot(hist['epoch'], hist['mae'], label='train error')
    plt.plot(hist['epoch'], hist['val_mae'], label='val error')
    plt.legend()
    plt.show()

        

model_plotted(history) 

#* Prediction
loss, mae, mse = model.evaluate(normed_test_data, test_label, verbose=0)
test_prediction = model.predict(normed_test_data).flatten()
# Plotting
plt.scatter(test_label, test_prediction)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
_ = plt.plot([-100,100], [-100, 100])
plt.show()

"""
128 Neurons: loss: 4.7247e-04 - mae: 0.0107 - mse: 4.7247e-04
64  Neurons: loss: 4.4533e-04 - mae: 0.0098 - mse: 4.4533e-04
32  Neurons: loss: 3.9095e-04 - mae: 0.0104 - mse: 3.9095e-04
16  Neurons: loss: 4.6131e-04 - mae: 0.0111 - mse: 4.6131e-04
"""

