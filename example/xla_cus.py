import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_DUMP_GRAPH_PREFIX'] = '/tmp/tf_graphs/'
os.environ['XLA_FLAGS'] = "--xla_dump_hlo_as_text --xla_dump_to=/home/chenhao/projects/xla_dump"
import signal

import numpy as np
import tensorflow as tf

# PID = os.getpid()

# a = tf.constant([1.8, 2.2], dtype=tf.cus)

npa = [1,2,3,4,5,6]
# npa = np.arange(1, 10, 1.6) # 1, 2.6, 4.2, 5.8, 7.4, 10
a = tf.cast(npa, tf.float32)
# a = tf.cast(npa, tf.uint32)

@tf.function(experimental_compile=True)
def model_fn(a):
    b = tf.cast(a, tf.cus)
    # c = tf.add(b,b)
    # c = tf.matmul(tf.reshape(b, [2,3]), tf.reshape(b, [3,2]))
    # return tf.cond(tf.greater_equal(b,b)[0], lambda: tf.add(b,b), lambda: tf.subtract(b,b))
    # return tf.cast(c, tf.float32)
    return tf.reshape(b, [2,3])

print(model_fn(a))
