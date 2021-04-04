import tensorflow as tf
a=[[2]]
m=tf.matmul(a, a)
word =tf.constant("hello")
one=tf.constant(5)
print(f"{word} {m}")

ten=tf.Variable(10)
print(ten)
sum =tf.multiply(ten, one)
update1=ten.assign(sum)#assign in tf2 is different fm tf1
#update2=one.assign(ten)#cannot assign to a constant
print(f"this is {ten}")

init_operation=tf.initializer()
sess=tf.compat.v1.Session()
sess.run(init_operation)
print(sess.run(sum))

a=tf.compat.v1.placeholder(tf.float32)
b=a*2

done=sess.run(b, feed_dict={a:3.0})
print(b)
