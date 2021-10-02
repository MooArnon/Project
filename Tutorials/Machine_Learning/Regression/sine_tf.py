import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

N = 100
x = np.random.rand(N, 1)
y = np.sin(2*np.pi*x) + 0.4*(np.random.rand(N, 1))
plt.plot(x, y, ".")
plt.close()

def leru(x):
    return tf.where(x>=0, x, 0)

class MLP():
    def __init__(self, neuron=[1, 100, 100, 1], activation= [leru, leru, None]):
        self.w = []
        self.activation = activation
        for i in range(1, len(neuron)):
            self.w.append(tf.Variable(np.random.randn(neuron[i-1], neuron[i]))) # w  ## use normal distribution random
            self.w.append(tf.Variable(np.random.randn(neuron[i]))) # b  ## use normal distribution random
    def __call__(self, x):
        for i in range(0, len(self.w), 2):
            x = x @ self.w[i] + self.w[i+1]
            if self.activation[i // 2] is not None:
                x = self.activation[i // 2](x)
        return x

lr = 0.0001
model = MLP()
for epoch in range(7000):
    with tf.GradientTape() as t:
        loss = tf.reduce_mean((model(x) - y)**2)
    dw = t.gradient(loss, model.w)
    for i, w in enumerate(model.w):
        w.assign_sub(lr * dw[i])
    if epoch == 1000:
        print(loss.numpy())

z = model(x)
plt.plot(x, z, ".r")
plt.plot(x, y, "." )
plt.show()

        

    
        
    


