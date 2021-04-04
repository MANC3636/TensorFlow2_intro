import tensorflow as tf
"""not clear on what optimizer does"""
a=tf.Variable(0.0)

def myloss(x):
    loss =a*x
    return loss

tf.print("my loss is ", myloss(5))

loss=lambda:abs(myloss(5)-[25])#why is 25 in brackets?

optimizer=tf.optimizers.Adam(1)#one is the learning rate
optimizer.minimize(loss, [a])#here, a is a trainable variable
tf.print(f"this is optimizer: {a}")

for i in range(1000):
    optimizer.minimize(loss,[a])

tf.print(f"this is optimizer: {a}")