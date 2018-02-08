import tensorflow as tf
import numpy as np

# === data setting

# [털, 날개]
x_data = np.array([
    [0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]
])
y_data = np.array([
    [1, 0, 0],  # 기타
    [0, 1, 0],  # 포유류
    [0, 0, 1],  # 조류
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

# === model 만들기

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 입력층 [특징, x]
# 출력측 [x, 분류 수]
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
W2 = tf.Variable(tf.random_uniform([10, 3], -1., 1.))

b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([3]))


L1 = tf.add(tf.matmul(X, W1), b1)
L = tf.nn.relu(L1)


model = tf.add(tf.matmul(L1, W2), b2)


# === 손실 함수 및 최적화

# 크로스 엔트로피
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

# 경사하강법 최적화
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)



# === 텐서플로우의 세션 초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# === 학습

# 위의 특징과 레이블 데이터를 이용해 학습을 100번 진행
for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    #학습 도중 10번에 한 번씩 손실값 출력
    if(step + 1) % 10 == 0:
        print(step+1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))



# === 학습 결과
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))