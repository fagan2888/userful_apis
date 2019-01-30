import tensorflow as tf
import numpy as np

batch_size = 2  # 批处理大小

hidden_size = 3  # 隐藏层神经元

max_time = 5  # 最大时间步长

depth = 6  # 输入层神经元数量，如词向量维度

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
basic_rnn_input_one_step = tf.Variable(tf.random_normal([batch_size, depth]))
initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

basic_rnn_input_steps = tf.Variable(tf.random_normal([batch_size, max_time, depth]))

# 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
# defining initial state
# 'state' is a tensor of shape [batch_size, cell_state_size]

dynamic_basic_rnn_outputs, dynamic_baisc_rnn_states = tf.nn.dynamic_rnn(rnn_cell, basic_rnn_input_steps,
                                                                        initial_state=initial_state, dtype=tf.float32)
baisc_rnn_output, basic_rnn_state = rnn_cell.call(basic_rnn_input_one_step, state=initial_state)

outputs = tf.transpose(dynamic_basic_rnn_outputs,[1,0,2])
last_output = outputs[-1] ###取得最后一步的输出


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(dynamic_basic_rnn_outputs))
    print(sess.run(dynamic_baisc_rnn_states))

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size)
initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

dynamic_lstm_inputs = tf.placeholder(np.float32, shape=(batch_size, max_time, depth))  # 32 是 batch_size
lstm_input_one_step_input = tf.Variable(tf.random_normal([batch_size, depth]))

dynamic_lstm_h0 = lstm_cell.zero_state(batch_size, np.float32)  # 通过zero_state得到一个全0的初始状态
outputs, state = tf.nn.dynamic_rnn(lstm_cell, inputs=dynamic_lstm_inputs, initial_state=initial_state)
lstm_one_step_output, lstm_one_step_state = lstm_cell.call(lstm_input_one_step_input, dynamic_lstm_h0)
print(lstm_one_step_output)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(outputs, feed_dict={dynamic_lstm_inputs: np.random.rand(batch_size, max_time, depth)}))
    last_output_=sess.run(last_output, feed_dict={dynamic_lstm_inputs: np.random.rand(batch_size, max_time, depth)})
    print(last_output_.shape)




import tensorflow as tf
import numpy as np

batch_size=5
num_units=64
num_steps=10
input_dim=8

input=np.random.randn(batch_size,num_steps,input_dim)
input[1,6:]=0
x=tf.placeholder(dtype=tf.float32,shape=[batch_size,num_steps,input_dim],name='input_x')
lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units,name='new')
initial_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)
y=tf.unstack(x,axis=1)
print(len(y))
# x:[batch_size,num_steps,input_dim],type:placeholder
# y:[num_steps,batch_size,input_dim],type:list
output,state=tf.nn.static_rnn(lstm_cell,y,initial_state=initial_state)
with tf.Session() as sess:
    init_op=tf.initialize_all_variables()
    sess.run(init_op)

    np.set_printoptions(threshold=np.NAN)

    result1,result2=(sess.run([output,state],feed_dict={x:input}))
    result1=np.asarray(result1)
    result2=np.asarray(result2)
    print(result1.shape)
    print(result2.shape)

import tensorflow as tf

x = tf.Variable(tf.random_normal([2, 4, 3]))  # [batch_size,timesteps,embedding_dim]

x = tf.unstack(x, axis=1)  # 按时间步展开
print(x)

n_neurons = 5  # 输出神经元数量

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons,name='new_cell')

output_seqs, states = tf.nn.static_rnn(basic_cell, x, dtype=tf.float32)

print(len(output_seqs))  # 四个时间步

print(output_seqs[0])  # 每个时间步输出一个张量

print(output_seqs[1])  # 每个时间步输出一个张量

print(states)  # 隐藏状态
