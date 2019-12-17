import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_DUMP_GRAPH_PREFIX'] = '/tmp/tf_graphs/'
os.environ['XLA_FLAGS'] = '--xla_dump_hlo_as_text --xla_dump_to=/tmp/xla/custom'

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

inp = np.random.rand(10, 10)

def model_fn(x):
    y = tf.sigmoid(tf.matmul(x, x))
    return y

x = tf.compat.v1.placeholder(tf.float32, shape=(10, 10))
[y] = tf.xla.experimental.compile(model_fn, inputs=[x])

with tf.compat.v1.Session() as sess:
    print(sess.run(y, feed_dict={x: inp}))
