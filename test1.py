import tensorflow as tf

# 그래프 생성
hello = tf.constant('Hello, TensorFlow!')
print(hello)

a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a, b)
print(c)

# 그래프 실행
sess = tf.Session()

print(sess.run(hello))
print(sess.run([a, b, c]))

sess.close()
