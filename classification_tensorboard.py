import tensorflow as tf
import numpy as np

# === data setting

# 데이터 파일 읽어오기
data = np.loadtxt('./data.csv', delimiter=',', unpack=True, dtype='float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

# === model 만들기

global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 묶은 블록은 텐서보드에서 한 계층 내부를 표현해줌
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.), name='W1')
    L1 = tf.nn.relu(tf.matmul(X, W1))
with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.), name='W2')
    L2 = tf.nn.relu(tf.matmul(L1, W2))
with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.), name='W3')
    model = tf.matmul(L2, W3)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost, global_step=global_step)

# 손실값 추적을 위해 수집할 값을 지정
# scalar 함수는 값이 하나인 텐서를 수집
# histogram, image, audio 등 다양한 값 수집 가능
tf.summary.scalar('cost', cost)

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model')

if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

# merged_all 함수로 앞서 지정한 텐서들을 수집
merged = tf.summary.merge_all()

# FileWriter 함수를 이용해 그래프와 텐서들의 값을 저장할 디렉터리 설정
writer = tf.summary.FileWriter('./logs', sess.graph)


# === 학습
for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    print('Step: %d, ' % sess.run(global_step),
          'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    # 모아둔 텐서의 값들을 계산하여 수집
    summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})

    # add_summary 함수를 이용해 해당 값들을 앞서 지정한 디렉터리에 자장
    # 나중에 확인할 수 있도록 global_step 값을 이용해 수집한 시점을 기록
    writer.add_summary(summary, global_step=sess.run(global_step))

saver.save(sess, './model/dnn.ckpt', global_step=global_step)

# === 학습 결과
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))

# terminal에서
# tensorboard --logdir=./logs 실행

# 각 가중치와 편향 등의 변화를 그래프로 살펴보고 싶다면
# tf.summary.histogram("Weights", W1) 실행

# TensorFlow Serving을 이용해 모델을 실제 서비스에 적용 가능