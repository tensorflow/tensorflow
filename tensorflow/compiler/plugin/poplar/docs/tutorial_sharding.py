import tensorflow as tf
import numpy as np

from tensorflow.contrib import ipu
from tensorflow.contrib.ipu.python.ops import ipu_scope
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

NUM_IPUS = 4

# Configure the IPU system
cfg = ipu.utils.create_ipu_config(profiling=True, use_poplar_text_report=True)
cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
cfg = ipu.utils.auto_select_ipus(cfg, NUM_IPUS)
ipu.utils.configure_ipu_system(cfg)


# Create the CPU section of the graph
with tf.device("cpu"):
    pa = tf.placeholder(np.float32, [2], name="a")
    pb = tf.placeholder(np.float32, [2], name="b")
    pc = tf.placeholder(np.float32, [2], name="c")


# Define a trace event
with tf.device('cpu'):
    report = gen_ipu_ops.ipu_event_trace()


# Distribute the computation across four shards
def sharded_graph(pa, pb, pc):
    with ipu.ops.ipu_shard(0):
        o1 = pa + pb
    with ipu.ops.ipu_shard(1):
        o2 = pa + pc
    with ipu.ops.ipu_shard(2):
        o3 = pb + pc
    with ipu.ops.ipu_shard(3):
        out = o1 + o2 + o3
        return out


# Create the IPU section of the graph
with ipu_scope("/device:IPU:0"):
    result = ipu.ipu_compiler.compile(sharded_graph, [pa, pb, pc])

with tf.Session() as sess:
    # sharded run
    result = sess.run(result, feed_dict={pa: [1., 1.], pb: [0., 1.], pc: [1., 5.]})

    print(result)