import tensorflow as tf

max_gradient_norm = 2
w1 = tf.Variable([[3.0, 2.0]])
params = tf.trainable_variables()
res = tf.matmul(w1, [[3.0], [1.]])
loss = (res - 2) ** 2
opt = tf.train.AdamOptimizer(0.1)
grads = tf.gradients(loss, params)
clipped_gradients, gradient_norm = tf.clip_by_global_norm(grads, max_gradient_norm)
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = opt.apply_gradients(zip(clipped_gradients, params), global_step=global_step)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(100):
        sess.run(train_op)
        print(sess.run([loss, res]))
        print(sess.run(clipped_gradients))
