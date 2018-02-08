import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# W, b 초기화
# -1.0 ~ 1.0 사이 균등분포 무작위 값
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# 자료 입력받을 플레이스홀더 설정
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

hypothesis = W * X + b

# 예측값과 실제값의 거리
# 모든 데이터에 대한 손실값의 평균
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 경사하강법(gradient descent)
# 하이퍼파라미터 => learning_rate
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(100):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

        print(step, cost_val, sess.run(W), sess.run(b))

    print("\n=== Test ===")
    print("X: 5, Y: ", sess.run(hypothesis, feed_dict={X: 5}))
    print("X: 2.5, Y: ", sess.run(hypothesis, feed_dict={X: 2.5}))