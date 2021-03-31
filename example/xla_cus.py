# # import os
# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['TF_DUMP_GRAPH_PREFIX'] = '/tmp/tf_graphs/'
# import signal

# import numpy as np
# import tensorflow as tf

# # PID = os.getpid()
# # os.kill(PID, signal.SIGUSR1)


# # a = tf.constant([1.8, 2.2], dtype=tf.cus)

# # a = tf.cast(np.arange(1,10), tf.bfloat16)
# a = tf.cast(np.arange(1,10), tf.cus)

# # b = tf.cast(np.arange(1,10), tf.bfloat16)
# b = tf.cast(np.arange(1,10), tf.cus)

# # inp = np.random.rand(10, 10)

# @tf.function(experimental_compile=True)
# def model_fn(a, b):
#     res = tf.add(a, b)
#     return res

# print(model_fn(a, b))




import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_DUMP_GRAPH_PREFIX'] = '/tmp/tf_graphs/'
os.environ['XLA_FLAGS'] = "--xla_dump_hlo_as_text --xla_dump_to=/mnt/xla_dump"
import signal

import numpy as np
import tensorflow as tf

# PID = os.getpid()

# a = tf.constant([1.8, 2.2], dtype=tf.cus)


npa = np.arange(1, 10, 1.3) # 1, 2.3, 3.6, 4.9, 6.2, 7.5, 8.8
a = tf.cast(npa, tf.float32)
# a = tf.cast(npa, tf.uint32)

@tf.function(experimental_compile=True)
def model_fn(a):
    b = tf.cast(a, tf.cus)
    c = tf.add(b, b)
    return c
    # return tf.cast(c, tf.float32)

print(model_fn(a))
