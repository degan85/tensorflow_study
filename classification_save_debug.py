import tensorflow as tf
import numpy as np

# === data setting

# 데이터 파일 읽어오기
data = np.loadtxt('./data.csv', delimiter=',', unpack=True, dtype='float32')

# 1열과 2열은 x_data
# 3열부터 마지막 열까지 y_data
# transpose 행과 열을 교환
# [[0,1,1,0,0,0,],[0,0,1,0,0,1]] => [[0.0],[1,0],[1,1],[0,0],[0,0],[0,0]]
x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

# === model 만들기

# 학습 횟수 카운트 변수
# 학습에 사용하지 않아서 trainale을 False
global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))
model = tf.matmul(L2, W3)


# === 손실 함수 및 최적화

# 크로스 엔트로피
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

# 경사하강법 최적화
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

# 최적화 함수가 최적화할 때마다 global_step 변수의 값이 1씩 증가 됨
train_op = optimizer.minimize(cost, global_step=global_step)

sess = tf.Session()

# 앞서 정의한 변수들을 가져오는 함수
# 변수들을 파일에 저장하거나
# 이전에 학습한 결과를불러와 담는 변수로 사용
saver = tf.train.Saver(tf.global_variables())

# 체크포인트 파일
ckpt = tf.train.get_checkpoint_state('./model')

# '/model 디렉터리에 기존에 학습해둔 모델이 있는지 확인
# 모델이 있다면 saver.restore 함수 사용 학습된 값들을 불러옴
# 아니면 변수를 초기화
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

# === 학습

for step in range(2):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    print('Step: %d, ' % sess.run(global_step),
          'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

# 최적화가 끝난 뒤 학습된 변수들을 저장한 체크포인트 파일에 저장
# './model' 디렉터리는 미리 생성돼 있어야 함
saver.save(sess, './model/dnn.ckpt', global_step=global_step)

# === 학습 결과
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))