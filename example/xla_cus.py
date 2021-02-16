import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_DUMP_GRAPH_PREFIX'] = '/tmp/tf_graphs/'

import numpy as np
import tensorflow as tf

# a = tf.constant([1.8, 2.2], dtype=tf.bfloat16)

# a = tf.constant([1.8, 2.2], dtype=tf.cus)
a = tf.cast(np.arange(1,10), tf.cus)

# b = tf.constant([1.8, 2.2], dtype=tf.cus)
b = tf.constant([1.8, 2.2], dtype=tf.bfloat16)

# b = tf.cast(np.arange(1,10), tf.cus)

# inp = np.random.rand(10, 10)

@tf.function(experimental_compile=True)
def model_fn(a, b):
    res = tf.add(a, b)
    return res

print(model_fn(a, b))
