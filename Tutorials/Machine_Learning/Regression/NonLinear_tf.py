from re import X
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.python.keras.api._v2.keras import callbacks, layers, optimizers
from tensorflow.python.keras.callbacks import History

#* Data processing
# Random data
n = 500
x = np.random.rand(n)
y = np.log(2*x) + 0.4*(np.random.rand(n))
# Build data frame
data = pd.DataFrame(x)
data['y'] = y
data = data.rename(columns={0:'x'})
# Split test and train
data_train = data.sample(frac=0.8, random_state=0)
data_test = data.drop(data_train.index)
train_label = data_train['y']
test_label = data_test['y']
plt.plot(data['x'], data['y'], '.')
plt.close()

#* Build model
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_dim=1),
        layers.Dense(32, activation=tf.nn.relu),
        layers.Dense(1)
        ])
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
    model.compile(loss='mae',
                   optimizer=optimizer,
                   metrics=['mse', 'mae'])

    return model

#*fit model
# stop fitting when model meet good condition
epoch = 2000
model = build_model()
stop_when_good = keras.callbacks.EarlyStopping(monitor='mse', patience=250)
history = model.fit(data_train['x'], train_label, 
                          epochs=epoch, validation_split=0.2, 
                          callbacks=stop_when_good)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

#* Measure how good of this model
def plot_model_mse():
    plt.xlabel('Epochs')
    plt.ylabel('Mean Square Error')
    plt.title('Mean Square Errors')
    plt.legend()
    plt.plot(hist['epoch'], hist['mse'])
    plt.plot(hist['epoch'], hist['val_mse'])
    plt.show()
def plot_model_mae():
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Errors')
    plt.legend()
    plt.plot(hist['epoch'], hist['mae'])
    plt.plot(hist['epoch'], hist['val_mae'])
    plt.show()


print(data_test)
#* Prediction
loss, mse, mae = model.evaluate(data_test['x'], test_label, verbose=0)
test_prediction = model.predict(data_test['x'])
# Plotting
plot_model_mse()
plot_model_mae()

plt.scatter(test_label, test_prediction)
plt.xlabel('True Values')
plt.ylabel('Predictions')
_ = plt.plot([-10,10], [-10, 10])
plt.show()

plt.plot(x, y, '.')
plt.plot(data_test['x'], test_prediction, '.r')
plt.show()


model.summary()
