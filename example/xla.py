import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_DUMP_GRAPH_PREFIX'] = '/tmp/tf_graphs/'

import numpy as np
import tensorflow as tf

inp = np.random.rand(10, 10)

@tf.function(experimental_compile=True)
def model_fn(x):
    y = tf.sigmoid(tf.matmul(x, x))
    return y

print(model_fn(inp))
