#* Easy code with one hidden layer
# Classify cloth from fashion_mnist 
import tensorflow as tf
from tensorflow.python.keras.api._v2.keras import metrics
data = tf.keras.datasets.fashion_mnist
(training_images, training_lables), (testing_images, testing_lables) = data.load_data()

training_images = training_images / 255.0
testing_images = testing_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

print(training_images)
