# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Naive LSTM to learn three-char time steps to one-char mapping
import numpy as np
from tensorflow.contrib import ipu
from tensorflow.contrib.ipu import utils
from tensorflow.contrib.ipu import popnn_rnn
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variables
from tensorflow.python.client import session as sl
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
from tensorflow.contrib.ipu import ipu_compiler

dataType = np.float32

seq_len = 3
batch_size = 40 - seq_len
input_size = 1
num_hidden = 64
num_training_steps = 100
lr = 10

def _PopnnLSTM(x, h, c, y):
  lstm_cell = popnn_rnn.PopnnLSTM(num_hidden,
    dtype=dataType,
    weights_initializer=init_ops.zeros_initializer(dtype=dataType),
    bias_initializer=init_ops.zeros_initializer(dtype=dataType))
  outputs, _ = lstm_cell(x, initial_state=(h, c), training=True)
  softmax = nn.softmax_cross_entropy_with_logits(logits=outputs[-1], labels=y)
  loss = math_ops.reduce_mean(softmax)
  train = gradient_descent.GradientDescentOptimizer(lr).minimize(loss)
  return [loss, train]

def _tfLSTM(x, h, c, y):
  lstm_cell = rnn_cell.LSTMCell(num_hidden,
    name='basic_lstm_cell',
    forget_bias=0.,
    initializer=init_ops.zeros_initializer(dtype=dataType))
  state = rnn_cell.LSTMStateTuple(c, h)
  outputs, states = rnn.dynamic_rnn(
      lstm_cell, x, dtype=dataType, initial_state=state, time_major=True)
  softmax = nn.softmax_cross_entropy_with_logits(logits=outputs[-1], labels=y)
  loss = math_ops.reduce_mean(softmax)
  train = gradient_descent.GradientDescentOptimizer(lr).minimize(loss)
  return [loss, train]

def _RunLayer(layer_func, x, y):
  with ops.device('cpu'):
    px = array_ops.placeholder(dataType, shape=x.shape)
    ph = array_ops.placeholder(dataType, shape=[batch_size, num_hidden])
    pc = array_ops.placeholder(dataType, shape=[batch_size, num_hidden])
    py = array_ops.placeholder(dataType, shape=y.shape)
  with ipu.ops.ipu_scope("/device:IPU:0"):
    r = ipu_compiler.compile(layer_func, inputs=[px, ph, pc, py])

  opts = utils.create_ipu_config(profiling=True, use_poplar_text_report=True)
  opts = ipu.utils.set_ipu_model_options(opts, compile_ipu_code=False)
  with sl.Session(config=config_pb2.ConfigProto(ipu_options=opts)) as sess:
    sess.run(variables.global_variables_initializer())
    fd = {px: x,
          ph: np.ones(ph.shape),
          pc: np.ones(pc.shape),
          py: y}
    losses = []
    for _ in range(0, num_training_steps):
      loss = sess.run(r, fd)
      losses.append(loss)
  return losses
def get_one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

class LstmSizeTest(test_util.TensorFlowTestCase):
  # Check that the loss goes down (and is identical to reference version).
  def testTraining(self):
    np.random.seed(42)
    nums = np.arange(batch_size + seq_len)
    # prepare the dataset of input to output pairs encoded as integers
    inputs = []
    one_hot = []
    for i in range(0, len(nums) - seq_len):
      sequence = nums[i:i + seq_len]
      output = nums[i + seq_len]
      inputs.append(sequence)
      one_hot.append(output)
    X = np.reshape(inputs, (seq_len, batch_size, input_size))
    # normalize
    X = X / float(len(nums))
    # one hot encode the output variable
    y = get_one_hot(nums[seq_len:], nums.size)
    labels = np.zeros([batch_size, num_hidden], dtype=dataType)
    labels[:y.shape[0],:y.shape[1]] = y

    custom_losses = _RunLayer(_PopnnLSTM, X, labels)
    # Check the loss goes down
    self.assertTrue(custom_losses[0] > custom_losses[-1])
    # Check that the loss is the same for the reference as well
    ref_losses = _RunLayer(_tfLSTM, X, labels)
    self.assertAllClose(custom_losses, ref_losses, atol=0.01)

if __name__ == "__main__":
    googletest.main()
