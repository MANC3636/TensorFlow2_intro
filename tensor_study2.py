import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

np.random.seed(42); tf.random.set_seed(42)


#random linear date; 100 bet 0 and 50
n=100
X=np.linspace(0,50, n)
y=np.linspace(0,50, n)

#adding noise to random linear data
X+=np.random.uniform(-10,10,n)
y+=np.random.uniform(-10,10,n)

#matplotlib to plot training data
plt.scatter(X,y)
plt.xlabel("x"); plt.ylabel("y");plt.title("training data")
plt.show()

"""linear model class"""
class LinearModel:
    def __init__(self):
        #y_pred =W*X +b
        self.W=tf.Variable(13.0)
        self.b=tf.Variable(4.0)


    def loss(self, y, y_pred):
        #print(self.W, self.b)
        return tf.reduce_mean(tf.square(y-y_pred))

    def train(self, X, y, lr=0.0001, epochs=20, verbose=True):
        def train_step():
            with tf.GradientTape() as t:
                current_loss = self.loss(y, self.predict(X))

            dW, db =t.gradient(current_loss, [self.W, self.b])
            self.W.assign_sub(lr*dW)#W-=lr*dW
            self.b.assign_sub(lr*db)#W-=lr*dW

            return current_loss
        for epoch in range(epochs):
            current_loss=train_step()
            if verbose:
                print(f"Epoch {epoch}: Loss: {current_loss.numpy()}")#<3 eager execution

    def predict(self, X):
        return self.W*X+self.b



model=LinearModel()
model.train(X, y)
plt.scatter(X, y, label ="data")
plt.plot(X, model.predict(X), "r-", label="predicted")
plt.legend()








