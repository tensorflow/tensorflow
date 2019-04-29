Retrieving a log
----------------

There are two mechanisms for retrieving the Poplar compilation/execution log. In
both cases the device must be configured to be an IPU model, and have
``profiling`` set to ``true``.

``ipu_event_trace()``
~~~~~~~~~~~~~~~~~~~~~

This is an op which retrieves all IPU events since the last time it was
executed. The operation must be placed on the CPU, and returns the events as a
one dimensional tensor of strings containing serialized IPU event protobufs,
from ``tensorflow.compiler.plugin.poplar.driver.trace_pb2.IpuTraceEvent``.

::

  import tensorflow as tf
  from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
  from tensorflow.core.protobuf import config_pb2
  from tensorflow.contrib.ipu import utils

  pa = tf.placeholder(tf.float32, [2,2], name="a")
  pb = tf.placeholder(tf.float32, [2,2], name="b")

  with tf.contrib.ipu.ops.ipu_scope("/device:IPU:0"):
    out = pa + pb

  with tf.device('cpu'):
    report = gen_ipu_ops.ipu_event_trace()

  opts = utils.create_ipu_config(profiling=True)
  with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as sess:
    result = sess.run(out, {pa:[[1,2],[3,4]], pb:[[5,6],[7,8]]})
    logs = sess.run(report)
    print tf.contrib.ipu.utils.extract_all_strings_from_event_trace(logs);


``ipu_compile_summary(name, op)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This produces a summary, which can be tied into the rest of the summary system
to produce output for Tensorboard. The parameter name is the name of the
summary, and op is one of the ops in the IPU graph. It is best to choose either
the inference output for an inference graph, the loss output for an evaluation
graph, or the train op for a training graph.

::

  import tensorflow as tf
  from tensorflow.contrib import ipu
  from tensorflow.core.protobuf import config_pb2

  ...

  tf.summary.scalar('c_out', c)
  ipu.ops.ipu_compile_summary('report', c)
  all_sum = tf.summary.merge_all()

  ...

  f = tf.summary.FileWriter('logs')
  with tf.Session() as s:
    sum_out, ... = s.run([add_sum, ...])
    f.add_summary(sum_out, 0)

    print "c = " + str(c)

