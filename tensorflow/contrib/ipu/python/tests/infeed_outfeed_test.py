from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.compiler import xla
from tensorflow.contrib import ipu
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.training import gradient_descent
from tensorflow.python.platform import googletest
from tensorflow.python.ops import variables
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.data.experimental.ops import sleep
from tensorflow.python.data.ops.iterator_ops import Iterator
from tensorflow.compiler.plugin.poplar.ops import gen_pop_datastream_ops
from threading import Thread



_N = 100
_BATCH_SIZE = 5
_N_ITER = _N//_BATCH_SIZE
_features = np.reshape(np.repeat(np.arange(_N), 10), (_N,10))

class InfeedOutfeedTest(test_util.TensorFlowTestCase):
  # non-commutative operation in loop to make sure tensors are 
  # pushed and popped in correct order
  def _loop_computation(batch_data, acc):
    x = math_ops.reduce_sum(batch_data)
    x = math_ops.divide(13.2, x)
    acc = math_ops.divide(x, acc)
    return acc

  _ground_truth = None
  def _get_ground_truth():
    if InfeedOutfeedTest._ground_truth is not None:
      return InfeedOutfeedTest._ground_truth

    feature_placeholder = array_ops.placeholder(np.float32, shape=(_features.shape))
    dataset_iter = InfeedOutfeedTest.create_dataset_iter(feature_placeholder)

    def cond(i, acc):
      return math_ops.less(i, _N_ITER)

    def body(i, acc):
      batch_data = dataset_iter.get_next()
      i = i + 1
      acc = InfeedOutfeedTest._loop_computation(batch_data, acc)
      return (i, acc)

    i = constant_op.constant(0)
    acc = constant_op.constant(1.0, dtype=np.float32)
    r = control_flow_ops.while_loop(cond, body, (i, acc), maximum_iterations=_N_ITER)

    with session_lib.Session() as sess:
      sess.run(dataset_iter.initializer,
          feed_dict={feature_placeholder: _features})
      (_, acc) = sess.run(r)
      InfeedOutfeedTest._ground_truth = acc

    return InfeedOutfeedTest._ground_truth


  def create_dataset_iter(feature_placeholder, sleep_in_pipeline=False):
    with ops.device('cpu'):
      dataset = Dataset.from_tensor_slices((feature_placeholder))
      if sleep_in_pipeline:
        # Sleep for 10000 microseconds to ensure interleaved execution of input pipeline thread
        # and graph execution thread. Simulates a context switch
        dataset = dataset.apply(sleep.sleep(10000))
      dataset = dataset.batch(batch_size=_BATCH_SIZE)
      dataset = dataset.repeat()
      dataset_iter = dataset.make_initializable_iterator()
      return dataset_iter

  def create_infeed_enqueue_while_loop(dataset_iter, feature_placeholder):
    def cond(j):
      return math_ops.less(j, _N_ITER)

    def body(j):
      batch_data = dataset_iter.get_next()
      enqueue_op = gen_pop_datastream_ops.pop_datastream_infeed_enqueue(batch_data)
      with ops.control_dependencies([enqueue_op]):
        j = j + 1
        return (j)

    j = constant_op.constant(0)
    enqueue_loop = control_flow_ops.while_loop(cond, body, (j,), maximum_iterations=_N_ITER)
    return enqueue_loop

  def my_net():
    def cond(i, acc):
      return math_ops.less(i, _N_ITER)

    def body(i, acc):
      i = i + 1
      batch_dequeued = gen_pop_datastream_ops.pop_datastream_infeed_dequeue(
        dtype=np.float32, shape=(_BATCH_SIZE,10))

      acc = InfeedOutfeedTest._loop_computation(batch_dequeued, acc)
      return (i, acc)

    i = constant_op.constant(0)
    acc = constant_op.constant(1.0, dtype=np.float32)
    r = control_flow_ops.while_loop(cond, body, (i, acc), maximum_iterations=_N_ITER)
    return r

  def testSequentialInfeedWhileLoop(self):
    ground_truth = InfeedOutfeedTest._get_ground_truth()
    feature_placeholder = array_ops.placeholder(np.float32, shape=(_features.shape))
    dataset_iter = InfeedOutfeedTest.create_dataset_iter(feature_placeholder)
    enqueue_loop = InfeedOutfeedTest.create_infeed_enqueue_while_loop(dataset_iter, feature_placeholder)

    with ipu.ops.ipu_scope("/device:IPU:0"):
      r = xla.compile(InfeedOutfeedTest.my_net)

    with session_lib.Session() as sess:
      sess.run(dataset_iter.initializer,
        feed_dict={feature_placeholder: _features})

      sess.run(enqueue_loop)

      (_, x) = sess.run(r)
      self.assertNear(x, ground_truth, 1e-6)


  def testMultiThreadedWhileLoop(self):
    ground_truth = InfeedOutfeedTest._get_ground_truth()
    feature_placeholder = array_ops.placeholder(np.float32, shape=(_features.shape))
    dataset_iter = InfeedOutfeedTest.create_dataset_iter(feature_placeholder, sleep_in_pipeline=True)
    enqueue_loop = InfeedOutfeedTest.create_infeed_enqueue_while_loop(dataset_iter, feature_placeholder)

    with ipu.ops.ipu_scope("/device:IPU:0"):
      r = xla.compile(InfeedOutfeedTest.my_net)

    def input_pipeline_thread_fn(sess):
      sess.run(enqueue_loop)

    def computation_thread_fn(sess, outputs):
      (_, x) = sess.run(r)
      outputs.append(x)


    # Run several times to heuristically check for race conditions
    for i in range(20):
      with session_lib.Session() as sess:
        sess.run(dataset_iter.initializer,
          feed_dict={feature_placeholder: _features})

        input_pipeline_thread = Thread(target=input_pipeline_thread_fn, args=(sess,))
        results = []
        computation_thread = Thread(target=computation_thread_fn, args=(sess, results))
        computation_thread.start()
        input_pipeline_thread.start()
          
        input_pipeline_thread.join()
        computation_thread.join()
          
        self.assertNear(results[0], ground_truth, 1e-6)




if __name__ == "__main__":
  googletest.main()
