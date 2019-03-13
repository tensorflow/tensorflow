# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.compiler.plugin.poplar.ops import gen_pop_datastream_ops

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.core.protobuf import config_pb2

import threading


class PopDatastreamTest(test_util.TensorFlowTestCase):
  def testSingleOutfeed(self):
    shape = [10,10]
    with ops.device("/device:IPU:0"):
      a = array_ops.placeholder(np.float32, shape)
      b = array_ops.placeholder(np.float32, shape)
      add = math_ops.add(a,b)
      outfeed_op = gen_pop_datastream_ops.pop_datastream_outfeed_enqueue([add])

    with ops.device('cpu'):
      outfeed = gen_pop_datastream_ops.pop_datastream_outfeed_dequeue(output_types=[np.float32], output_shapes=[shape])

    with session_lib.Session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(outfeed_op, feed_dict={a:np.ones(shape, np.float32), b:np.ones(shape, np.float32)})
      outfed = sess.run(outfeed)

      self.assertAllClose(outfed[0], 2*np.ones(shape, np.float32))



  def testTupleOutfeedGetAll(self):
    shape_1 = [10,10]
    shape_2 = [4, 4]

    with ops.device("/device:IPU:0"):
      a = array_ops.placeholder(np.float32, shape_1)
      b = array_ops.placeholder(np.float32, shape_1)
      c = array_ops.placeholder(np.float32, shape_2)
      d = array_ops.placeholder(np.float32, shape_2)
      add = math_ops.add(a,b)
      sub = math_ops.sub(c,d)
      outfeed_op = gen_pop_datastream_ops.pop_datastream_outfeed_enqueue([add, sub])

    with ops.device('cpu'):
      outfeed = gen_pop_datastream_ops.pop_datastream_outfeed_dequeue(output_types=[np.float32, np.float32], output_shapes=[None, shape_1, shape_2])

    def get_result(sess, result):
      result.append(sess.run(outfeed))

    with session_lib.Session() as sess:
      sess.run(variables.global_variables_initializer())
      result = []
      sess.run(outfeed_op, feed_dict={a:np.ones(shape_1, np.float32), b:np.ones(shape_1, np.float32), c:np.ones(shape_2, np.float32), d:np.ones(shape_2, np.float32) })
      sess.run(outfeed_op, feed_dict={a:2*np.ones(shape_1, np.float32), b:np.ones(shape_1, np.float32), c:2*np.ones(shape_2, np.float32), d:np.ones(shape_2, np.float32) })
      outfed = sess.run(outfeed)
      self.assertTrue(len(outfed) == 2)
      self.assertEqual(outfed[0].shape, (2,10,10))
      self.assertEqual(outfed[1].shape, (2,4,4))
      self.assertAllClose(outfed[0][0], np.broadcast_to(2, [10, 10]))
      self.assertAllClose(outfed[0][1], np.broadcast_to(3, [10, 10]))
      self.assertAllClose(outfed[1][0], np.broadcast_to(0, [4, 4]))
      self.assertAllClose(outfed[1][1], np.broadcast_to(1, [4, 4]))

  def testTupleOutfeedGetLast(self):
      shape_1 = [10,10]
      shape_2 = [4, 4]

      with ops.device("/device:IPU:0"):
        a = array_ops.placeholder(np.float32, shape_1)
        b = array_ops.placeholder(np.float32, shape_1)
        c = array_ops.placeholder(np.float32, shape_2)
        d = array_ops.placeholder(np.float32, shape_2)
        add = math_ops.add(a,b)
        sub = math_ops.sub(c,d)
        outfeed_op = gen_pop_datastream_ops.pop_datastream_outfeed_enqueue([add, sub], outfeed_mode='get_last')

      with ops.device('cpu'):
        outfeed = gen_pop_datastream_ops.pop_datastream_outfeed_dequeue(output_types=[np.float32, np.float32], output_shapes=[shape_1, shape_2])

      def get_result(sess, result):
        result.append(sess.run(outfeed))

      with session_lib.Session() as sess:
        sess.run(variables.global_variables_initializer())
        result = []
        sess.run(outfeed_op, feed_dict={a:np.ones(shape_1, np.float32), b:np.ones(shape_1, np.float32), c:np.ones(shape_2, np.float32), d:np.ones(shape_2, np.float32) })
        sess.run(outfeed_op, feed_dict={a:2*np.ones(shape_1, np.float32), b:np.ones(shape_1, np.float32), c:2*np.ones(shape_2, np.float32), d:np.ones(shape_2, np.float32) })
        outfed = sess.run(outfeed)
        self.assertTrue(len(outfed) == 2)
        self.assertEqual(outfed[0].shape, (10,10))
        self.assertEqual(outfed[1].shape, (4,4))
        self.assertAllClose(outfed[0], np.broadcast_to(3, [10, 10]))
        self.assertAllClose(outfed[1], np.broadcast_to(1, [4, 4]))


  def testOutfeedGetAll(self):
    shape = [2,2]
    with ops.device("/device:IPU:0"):
      a = array_ops.placeholder(np.float32, shape)
      b = array_ops.placeholder(np.float32, shape)
      add = math_ops.add(a,b)
      outfeed_op = gen_pop_datastream_ops.pop_datastream_outfeed_enqueue([add], outfeed_mode='all')

    with ops.device('cpu'):
      outfeed_all = gen_pop_datastream_ops.pop_datastream_outfeed_dequeue(output_types=[np.float32], output_shapes=[None, shape])

    with session_lib.Session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(outfeed_op, feed_dict={a:np.ones(shape, np.float32), b:np.ones(shape, np.float32)})
      sess.run(outfeed_op, feed_dict={a:3.1*np.ones(shape, np.float32), b:2*np.ones(shape, np.float32)})

      outfed = sess.run(outfeed_all)
      self.assertTrue(len(outfed) == 1)
      self.assertEqual(outfed[0].shape, (2, 2, 2))
      self.assertAllClose(outfed[0][0], 2*np.ones(shape, np.float32))
      self.assertAllClose(outfed[0][1], (3.1+2)*np.ones(shape, np.float32))


  def testOutfeedGetLast(self):
    shape = [2,2]
    with ops.device("/device:IPU:0"):
      a = array_ops.placeholder(np.float32, shape)
      b = array_ops.placeholder(np.float32, shape)
      add = math_ops.add(a,b)
      outfeed_op = gen_pop_datastream_ops.pop_datastream_outfeed_enqueue([add], outfeed_mode='get_last')

    with ops.device('cpu'):
      outfeed_last = gen_pop_datastream_ops.pop_datastream_outfeed_dequeue(output_types=[np.float32], output_shapes=[shape])

    with session_lib.Session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(outfeed_op, feed_dict={a:np.ones(shape, np.float32), b:np.ones(shape, np.float32)})
      sess.run(outfeed_op, feed_dict={a:3.1*np.ones(shape, np.float32), b:2*np.ones(shape, np.float32)})

      outfed = sess.run(outfeed_last)
      self.assertTrue(len(outfed) == 1)
      self.assertEqual(outfed[0].shape, (2, 2))
      self.assertAllClose(outfed[0], (3.1+2)*np.ones(shape, np.float32))




if __name__ == "__main__":
  googletest.main()
