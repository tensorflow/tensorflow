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
from tensorflow.contrib.compiler import xla
from tensorflow.contrib import ipu
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.core.protobuf import config_pb2

from tensorflow.contrib.ipu import loops
from tensorflow.contrib.ipu import ipu_infeed_queue

def create_increasing_dataset(value, shape=[4,4], dtype=np.float32):
  def _get_one_input(data):
    return math_ops.cast(
            gen_array_ops.broadcast_to(data, shape=shape), dtype=dtype)

  dataset = Dataset.range(value).repeat().map(_get_one_input)
  return dataset

class InfeedOutfeedTest(test_util.TensorFlowTestCase):

  def testSingleInfeedRepeatNonTuple(self):
    dataset = create_increasing_dataset(10)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)

    def body(v):
      v = v + infeed_queue.get_next()
      return (v)

    def my_net(v):
      r = loops.repeat(20, body, (v), infeed_queue)
      return r

    with ops.device('cpu'):
      v = array_ops.placeholder(np.float32, [4, 4])

    with ipu.ops.ipu_scope("/device:IPU:0"):
      res = xla.compile(my_net, inputs=[v])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    with session_lib.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res, {v:np.ones([4, 4], np.float32)})
      self.assertAllClose(result[0], np.broadcast_to(91, [4, 4]))

  def testSingleInfeedRepeatTuple(self):
    dataset = create_increasing_dataset(3)

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return (image_1, image_2)
    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)

    def body(v):
      im1, im2 = infeed_queue.get_next()
      v = v + im1 + im2
      return (v)

    def my_net():
      v = constant_op.constant(0.0, shape=[4,4], dtype=np.float32)
      r = loops.repeat(5, body, [v], infeed_queue)
      return r

    with ipu.ops.ipu_scope("/device:IPU:0"):
      res = xla.compile(my_net, inputs=[])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    with session_lib.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(31, [4, 4]))

  def testSingleInfeedRepeatNamed(self):
    dataset = create_increasing_dataset(3)

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return {"a": image_1,
              "b": image_2}
    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)

    def body(v):
      data = infeed_queue.get_next()
      v = v + data["a"] + data["b"]
      return (v)

    def my_net():
      v = constant_op.constant(0.0, shape=[4,4], dtype=np.float32)
      r = loops.repeat(5, body, [v], infeed_queue)
      return r

    with ipu.ops.ipu_scope("/device:IPU:0"):
      res = xla.compile(my_net, inputs=[])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    with session_lib.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(31, [4, 4]))

  def testSingleInfeedRepeatMultipleDequeues(self):
    dataset = create_increasing_dataset(2)

    def dataset_parser(value):
      image_1 = value + 1
      image_2 = image_1 * 2
      return {"a": image_1,
              "b": image_2}
    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)

    # Note how we get the value for a from the first dequeue and value for b
    # from the second dequeue.
    def body(v):
      v = v + infeed_queue.get_next()["a"] + infeed_queue.get_next()["b"]
      return (v)

    def my_net():
      v = constant_op.constant(0.0, shape=[4,4], dtype=np.float32)
      r = loops.repeat(5, body, [v], infeed_queue)
      return r

    with ipu.ops.ipu_scope("/device:IPU:0"):
      res = xla.compile(my_net, inputs=[])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    with session_lib.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
      sess.run(infeed_queue.initializer)
      with self.assertRaisesRegexp(errors.FailedPreconditionError,
                                   'Currently calling'):
        sess.run(res)

  def testSingleInfeedMultipleRepeats(self):
    dataset = create_increasing_dataset(2)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)

    # Note how we get the value for a from the first dequeue and value for b
    # from the second dequeue.
    def body(v):
      v = v + infeed_queue.get_next()
      return (v)

    def my_net():
      v = constant_op.constant(0.0, shape=[4,4], dtype=np.float32)
      r = loops.repeat(5, body, [v], infeed_queue)
      r = loops.repeat(5, body, [r], infeed_queue)
      return r

    with ipu.ops.ipu_scope("/device:IPU:0"):
      res = xla.compile(my_net, inputs=[])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    with session_lib.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res)
      self.assertAllClose(result[0], np.broadcast_to(5, [4, 4]))

  def testSingleInfeedWhileLoopNonTuple(self):
    dataset = create_increasing_dataset(10)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)

    def cond(i, v):
      return i < 20

    def body(i, v):
      v = v + infeed_queue.get_next()
      return (i + 1, v)

    def my_net(v):
      i = 0
      r = loops.while_loop(cond, body, (i, v), infeed_queue)
      return r[1]

    with ops.device('cpu'):
      v = array_ops.placeholder(np.float32, [4, 4])

    with ipu.ops.ipu_scope("/device:IPU:0"):
      res = xla.compile(my_net, inputs=[v])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    with session_lib.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res, {v:np.ones([4, 4], np.float32)})
      self.assertAllClose(result[0], np.broadcast_to(91, [4, 4]))

  def testSingleInfeedWhileLoopTuple(self):
    dataset = create_increasing_dataset(3)

    def dataset_parser(value):
      image_1 = value
      image_2 = (value + 10.) / 2.0
      return (image_1, image_2)
    dataset = dataset.map(dataset_parser)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)

    def cond(i, v):
      return i < 20

    def body(i, v):
      im1, im2 = infeed_queue.get_next()
      v = v + im1 + im2
      return (i + 1, v)

    def my_net(v):
      i = 0
      r = loops.while_loop(cond, body, (i, v), infeed_queue)
      return r[1]

    with ops.device('cpu'):
      v = array_ops.placeholder(np.float32, [4, 4])

    with ipu.ops.ipu_scope("/device:IPU:0"):
      res = xla.compile(my_net, inputs=[v])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    with session_lib.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(res, {v:np.ones([4, 4], np.float32)})
      self.assertAllClose(result[0], np.broadcast_to(129.5, [4, 4]))

  def testSingleInfeedMultipleRuns(self):
    dataset = create_increasing_dataset(10)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)

    def program(iters):
      def body(v):
        v = v + infeed_queue.get_next()
        return (v)

      def my_net():
        v = constant_op.constant(0.0, shape=[4,4], dtype=np.float32)
        r = loops.repeat(iters, body, (v), infeed_queue)
        return r

      with ipu.ops.ipu_scope("/device:IPU:0"):
        return xla.compile(my_net)

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    with session_lib.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
      sess.run(infeed_queue.initializer)
      result = sess.run(program(0))
      self.assertAllClose(result[0], np.broadcast_to(0, [4, 4]))
      # The iterator has not moved - next element should be all 1s.
      result = sess.run(program(2))
      self.assertAllClose(result[0], np.broadcast_to(1, [4, 4]))
      # The iterator has moved - in the next two iterations it should pull 2 and 3.
      result = sess.run(program(2))
      self.assertAllClose(result[0], np.broadcast_to(5, [4, 4]))
      # The iterator has moved - in the next two iterations it should pull 4 and 5.
      result = sess.run(program(2))
      self.assertAllClose(result[0], np.broadcast_to(9, [4, 4]))

  def testTwoInfeedsDifferentPrograms(self):
    dataset1 = create_increasing_dataset(20)
    dataset2 = create_increasing_dataset(3)

    infeed_queue1 = ipu_infeed_queue.IPUInfeedQueue(dataset1)
    infeed_queue2 = ipu_infeed_queue.IPUInfeedQueue(dataset2)

    def program(iters, infeed_queue):
      def body(v):
        v = v + infeed_queue.get_next()
        return (v)

      def my_net():
        v = constant_op.constant(0.0, shape=[4,4], dtype=np.float32)
        r = loops.repeat(iters, body, (v), infeed_queue)
        return r

      with ipu.ops.ipu_scope("/device:IPU:0"):
        return xla.compile(my_net)

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    with session_lib.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
      sess.run(infeed_queue1.initializer)
      sess.run(infeed_queue2.initializer)
      result = sess.run(program(5, infeed_queue1))
      self.assertAllClose(result[0], np.broadcast_to(10, [4, 4]))
      result = sess.run(program(5, infeed_queue2))
      self.assertAllClose(result[0], np.broadcast_to(4, [4, 4]))
      result = sess.run(program(5, infeed_queue1))
      self.assertAllClose(result[0], np.broadcast_to(35, [4, 4]))
      result = sess.run(program(5, infeed_queue2))
      self.assertAllClose(result[0], np.broadcast_to(5, [4, 4]))

  def testUndefinedShape(self):
    dataset = create_increasing_dataset(10)
    dataset = dataset.batch(10, drop_remainder=False)
    with self.assertRaisesRegexp(ValueError,
                                 'Output shape \(\?,'):
      infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)

  def testTrainingLoopWithInfeed(self):
    dataset = create_increasing_dataset(10, shape=[4,4,2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)

    def my_net(iters):
      def body(loss):
        x = infeed_queue.get_next()
        with variable_scope.variable_scope("vs", use_resource=True):
          y = convolutional.conv2d(x, 2, 1, use_bias=True,
                                 kernel_initializer=init_ops.ones_initializer(),
                                 name='conv1')
        loss = math_ops.reduce_sum(y)
        optimizer = gradient_descent.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)
        with ops.control_dependencies([train]):
          return array_ops.identity(loss)

      loss = 0.0
      return loops.repeat(iters, body, (loss), infeed_queue)

    with ops.device('cpu'):
      iters = array_ops.placeholder(np.int32, shape=[])

    with ipu.ops.ipu_scope("/device:IPU:0"):
      r = xla.compile(my_net, inputs=[iters])

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      sess.run(variables.global_variables_initializer())
      initial_loss = sess.run(r, {iters: 1})
      final_loss = sess.run(r, {iters: 1000})
      self.assertTrue(initial_loss > final_loss)

if __name__ == "__main__":
  googletest.main()
