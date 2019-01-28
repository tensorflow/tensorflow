from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
from tensorflow.python.client import session as session_lib
from tensorflow.compiler.plugin.poplar.ops import gen_pop_datastream_ops
from tensorflow.python.platform import googletest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.platform import test
from tensorflow.python.data import experimental
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.data.ops.iterator_ops import Iterator


class InfeedTest(test.TestCase):
  def testInfeed(self):
    features = -np.array(np.reshape(np.arange(100),(10,10)), dtype=np.float32)
    batch_size = 1
    feature_placeholder = array_ops.placeholder(np.float32, shape=(features.shape))

    dataset = Dataset.from_tensor_slices((feature_placeholder))
    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.repeat()
    dataset_iter = dataset.make_initializable_iterator()
    batch_data = dataset_iter.get_next()

    batch_data.set_shape((batch_size,) + features.shape[1:])
    with ops.device("/device:CPU:0"):
      enqueue_op = gen_pop_datastream_ops.pop_datastream_infeed_enqueue(batch_data)

    with ops.device("/device:IPU:0"):
      batch_dequeued = gen_pop_datastream_ops.pop_datastream_infeed_dequeue(dtype=batch_data.dtype,
        shape=(1,10))
      c = math_ops.abs(batch_dequeued)


    with session_lib.Session() as sess:
      sess.run(dataset_iter.initializer,
        feed_dict={feature_placeholder: features})
      for i in range(10):
        result = sess.run(enqueue_op)

      for i in range(10):
        output = sess.run(c)
        self.assertAllEqual(output, np.reshape(np.abs(features[i]), (1,10)))

  def testInfeedTuples(self):
    features = -np.array(np.reshape(np.arange(100),(10,10)), dtype=np.float32)
    batch_size = 1
    feature_placeholder = array_ops.placeholder(np.float32, shape=(features.shape))

    dataset = Dataset.from_tensor_slices((feature_placeholder))
    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.repeat()
    dataset_iter = dataset.make_initializable_iterator()
    batch_data = dataset_iter.get_next()


    with ops.device("/device:CPU:0"):
      x = (batch_data, batch_data)
      enqueue_op = gen_pop_datastream_ops.pop_datastream_infeed_enqueue_tuple(x, shapes=[(1,10), (1,10)])

    with ops.device("/device:IPU:0"):
      a, b = gen_pop_datastream_ops.pop_datastream_infeed_dequeue_tuple(dtypes=[batch_data.dtype, batch_data.dtype],
        shapes=[(1,10), (1,10)])
      c = math_ops.add(math_ops.abs(a), math_ops.abs(b))

    with session_lib.Session() as sess:
      sess.run(dataset_iter.initializer,
        feed_dict={feature_placeholder: features})
      for i in range(10):
        result = sess.run(enqueue_op)

      for i in range(10):
        output = sess.run(c)
        self.assertAllEqual(output, np.reshape(np.abs(2*features[i]), (1,10)))




if __name__ == '__main__':
  googletest.main()
