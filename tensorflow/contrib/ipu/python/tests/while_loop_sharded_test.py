# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib import ipu
from tensorflow.contrib.ipu.python import autoshard
from tensorflow.contrib.ipu.python import ipu_compiler
from tensorflow.contrib.ipu.python import ipu_infeed_queue
from tensorflow.contrib.ipu.python import loops
from tensorflow.contrib.ipu.python import sharded_optimizer as so
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import gradient_descent
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent as gd


def create_increasing_dataset(value,
                              data_shape=[1, 32, 32, 4],
                              label_shape=[1, 8],
                              dtype=np.float32):
  def _get_one_input(data):
    return (math_ops.cast(
        gen_array_ops.broadcast_to(data, shape=data_shape), dtype=dtype),
            math_ops.cast(
                gen_array_ops.broadcast_to(data, shape=label_shape),
                dtype=dtype))

  dataset = Dataset.range(value).repeat().map(_get_one_input)
  return dataset


class WhileLoopShardedTest(test_util.TensorFlowTestCase):
  def testSimpleXlaCompileTrainingInLoopWithParam(self):
    dataset = create_increasing_dataset(3)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)

    def my_net(lr):
      def my_model(lr, loss, x, y):
        with ipu.ops.ipu_scope("/device:IPU:0"):
          inp = x

          x = convolutional.conv2d(
              x, 8, 3, padding='same', name="conv1", use_bias=False)
          x = math_ops.reduce_max(x, axis=[1, 2])

          cross_entropy = nn.softmax_cross_entropy_with_logits(
              logits=x, labels=y)
          loss = math_ops.reduce_mean(cross_entropy)

          optim = so.ShardedOptimizer(gd.GradientDescentOptimizer(lr))
          train = optim.minimize(cross_entropy)

          autoshard.automatic_sharding(2, inp, loss, [])

          return [lr, loss, train]

      loss = 0.0
      return loops.repeat(2, my_model, [lr, loss], infeed_queue)

    lr = array_ops.placeholder(dtypes.float32, [])
    out = ipu_compiler.compile(my_net, inputs=[lr])

    cfg = ipu.utils.create_ipu_config(profiling=False)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    cfg = ipu.utils.auto_select_ipus(cfg, 2)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:
      sess.run(infeed_queue.initializer)
      sess.run(variables.global_variables_initializer())
      sess.run(out[0], {lr: 0.1})


if __name__ == "__main__":
  googletest.main()
