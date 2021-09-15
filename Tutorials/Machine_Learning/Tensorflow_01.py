#* Easy code with one hidden layer
# Classify cloth from fashion_mnist 
import tensorflow as tf
from tensorflow.python.keras.api._v2 import keras
from tensorflow.python.keras.api._v2.keras import layers, metrics
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.ops.gen_nn_ops import Conv2D, MaxPool
from tensorflow.python.ops.nn_ops import max_pool2d
data = tf.keras.datasets.fashion_mnist
(training_images, training_lables), (testing_images, testing_lables) = data.load_data()

# Reduce pixel size
training_images = training_images.reshape(60000, 28, 28, 1)
testing_images = testing_images.reshape(10000, 28, 28, 1) 
training_images = training_images / 255.0
testing_images = testing_images / 255.0

# Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Model Medthology
model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# Train model
model.fit(training_images, training_lables, epochs=50)

model.evaluate(testing_images, testing_lables)

classifications = model.predict(testing_images)
print(classifications[0])
print(testing_lables[0])
