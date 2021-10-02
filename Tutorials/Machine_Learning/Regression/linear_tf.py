import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

N = 100
x = np.random.rand(N)
y = 3 + 4*x + np.random.rand(N)
plt.scatter(x, y)
plt.close()

w = np.random.rand()
b = np.random.rand()

w = tf.Variable(w)
b = tf.Variable(b)
lr = 0.1

for epoch in range(1000):
    with tf.GradientTape() as t:
        Y = w*x + b
        loss = tf.reduce_mean((y - Y)**2) #! Use mean in tensorflow, not from numpy 
    dw, db = t.gradient(loss, [w, b]) # de/dw , de/db
    w.assign_sub(lr * dw) # w == lr * dw
    b.assign_sub(lr * db)
    print(epoch, w.numpy(), b.numpy(), loss.numpy())
z =w*x + b
plt.plot(x, y, ".")
plt.plot(x, z, ".g")
plt.show()
