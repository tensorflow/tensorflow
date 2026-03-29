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
# ==============================================================================
"""Tests for gradients of (block) LSTM/GRU operations."""

import functools

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_rnn_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.framework import errors


class RNNGradTest(test.TestCase):

  @test_util.deprecated_graph_mode_only
  def testBlockLSTMV1V2Consistency(self):
    num_steps = 1
    batch_size = 1
    input_size = 1
    hidden_size = 8
    w = deterministic_random_uniform(
        [input_size + hidden_size, 4 * hidden_size])
    b = deterministic_random_uniform([4 * hidden_size])
    x = deterministic_random_uniform([num_steps, batch_size, input_size])
    cs_prev = h_prev = deterministic_random_uniform([batch_size, hidden_size])

    all_cs, all_h = self._lstm_block(
        functools.partial(
            gen_rnn_ops.BlockLSTM,
            forget_bias=0.0,  # Disable to match V2 default.
            cell_clip=0.0),  # Disable to match V2 default.
        w, b, x, cs_prev, h_prev)
    w_grad, b_grad = gradients.gradients(all_cs + all_h, [w, b])

    w_ifco, b_ifco = icfo_to_ifco(w, b)
    all_cs_ifco, all_h_ifco = self._lstm_block(
        gen_rnn_ops.BlockLSTMV2, w_ifco, b_ifco, x, cs_prev, h_prev)
    w_ifco_grad, b_ifco_grad = gradients.gradients(
        all_cs_ifco + all_h_ifco, [w_ifco, b_ifco])

    self.assertAllEqual(all_cs, all_cs_ifco)
    self.assertAllEqual(all_h, all_h_ifco)
    self.assertAllEqual(w_grad, w_ifco_grad)
    self.assertAllEqual(b_grad, b_ifco_grad)

  @test_util.deprecated_graph_mode_only
  def testLSTMBlockCell(self):
    batch_size = np.random.randint(1, 32)
    input_size = np.random.randint(1, 32)
    hidden_size = np.random.randint(1, 32)
    w = deterministic_random_uniform(
        [input_size + hidden_size, 4 * hidden_size])
    b = deterministic_random_uniform([4 * hidden_size])
    x = deterministic_random_uniform([batch_size, input_size])
    cs_prev = h_prev = deterministic_random_uniform([batch_size, hidden_size])
    w_peephole = array_ops.zeros(cs_prev.shape[1:], dtype=w.dtype)
    cs_grad = deterministic_random_uniform([batch_size, hidden_size])
    h_grad = deterministic_random_uniform([batch_size, hidden_size])

    outputs = []
    grads = []
    for use_gpu in [False, True]:
      with self.cached_session(use_gpu=use_gpu):
        output = gen_rnn_ops.lstm_block_cell(
            x=x,
            cs_prev=cs_prev,
            h_prev=h_prev,
            w=w,
            wci=w_peephole,
            wcf=w_peephole,
            wco=w_peephole,
            b=b,
            forget_bias=1.0,
            cell_clip=0.0,
            use_peephole=False)
        (i, cs, f, o, ci, co, _) = output
        grad = gen_rnn_ops.lstm_block_cell_grad(
            x=x,
            cs_prev=cs_prev,
            h_prev=h_prev,
            w=w,
            wci=w_peephole,
            wcf=w_peephole,
            wco=w_peephole,
            b=b,
            i=i,
            cs=cs,
            f=f,
            o=o,
            ci=ci,
            co=co,
            cs_grad=cs_grad,
            h_grad=h_grad,
            use_peephole=False)
        outputs.append(output)
        grads.append(grad)
    self.assertAllClose(outputs[0], outputs[1])
    self.assertAllClose(grads[0], grads[1])

  def _lstm_block(self, op, w, b, x, cs_prev, h_prev):
    w_peephole = array_ops.zeros(cs_prev.shape[1:], dtype=w.dtype)
    _, all_cs, _, _, _, _, all_h = op(
        seq_len_max=math_ops.cast(array_ops.shape(x)[0], dtypes.int64),
        x=x,
        cs_prev=cs_prev,
        h_prev=h_prev,
        w=w,
        wci=w_peephole,
        wcf=w_peephole,
        wco=w_peephole,
        b=b,
        use_peephole=False)
    return all_cs, all_h

  def testBlockLSTMGradV2TimeDimValidation(self):
    seq_len_max = 3
    batch_size = 2
    input_size = 4
    num_units = 5

    x = array_ops.constant(
      np.random.randn(
        seq_len_max - 1, batch_size, input_size
      ).astype(np.float32)
    ) # x: (2,2,4), seq_len_max:3
    cs_prev = array_ops.constant(
        np.random.randn(batch_size, num_units).astype(np.float32)
    )
    h_prev = array_ops.constant(
      np.random.randn(
          batch_size, num_units
      ).astype(np.float32)
    )
    w = array_ops.constant(
      np.random.randn(
          input_size + num_units, 4 * num_units
        ).astype(np.float32)
    )
    wci = array_ops.constant(
      np.random.randn(
        num_units
      ).astype(np.float32)
    )
    wcf = array_ops.constant(np.random.randn(num_units).astype(np.float32))
    wco = array_ops.constant(np.random.randn(num_units).astype(np.float32))
    b = array_ops.constant(np.random.randn(4 * num_units).astype(np.float32))
    i = array_ops.constant(
      np.random.randn(
        seq_len_max - 1, batch_size, num_units
      ).astype(np.float32)
    )
    cs = array_ops.constant(
      np.random.randn(
        seq_len_max - 1, batch_size, num_units
      ).astype(np.float32)
    )
    f = array_ops.constant(
      np.random.randn(
        seq_len_max - 1, batch_size, num_units
      ).astype(np.float32)
    )
    o = array_ops.constant(
      np.random.randn(
        seq_len_max - 1, batch_size, num_units
      ).astype(np.float32)
    )
    ci = array_ops.constant(
      np.random.randn(
        seq_len_max - 1, batch_size, num_units
      ).astype(np.float32)
    )
    co = array_ops.constant(
      np.random.randn(
        seq_len_max - 1, batch_size, num_units
      ).astype(np.float32)
    )
    h = array_ops.constant(
      np.random.randn(
        seq_len_max - 1, batch_size, num_units
      ).astype(np.float32)
    )
    cs_grad = array_ops.constant(
      np.random.randn(
        seq_len_max - 1, batch_size, num_units
      ).astype(np.float32)
    )
    h_grad = array_ops.constant(
      np.random.randn(
        seq_len_max - 1, batch_size, num_units
      ).astype(np.float32)
    )
    use_peephole = True

    with self.assertRaisesRegex(
                                errors.InvalidArgumentError, "time dimension"):
      result = gen_rnn_ops.block_lstm_grad_v2(
            seq_len_max=seq_len_max, # seq_len_max:3
            x=x, #  x: (2,2,4)
            cs_prev=cs_prev,
            h_prev=h_prev,
            w=w,
            wci=wci,
            wcf=wcf,
            wco=wco,
            b=b,
            i=i,
            cs=cs,
            f=f,
            o=o,
            ci=ci,
            co=co,
            h=h,
            cs_grad=cs_grad,
            h_grad=h_grad,
            use_peephole=use_peephole
        )


def deterministic_random_uniform(shape):
  return ops.convert_to_tensor(np.random.random(shape), dtype=dtypes.float32)


def icfo_to_ifco(w, b):
  """Convert gates' weights and biases from ICFO to IFCO layout."""
  w_i, w_c, w_f, w_o = array_ops.split(w, num_or_size_splits=4, axis=1)
  b_i, b_c, b_f, b_o = array_ops.split(b, num_or_size_splits=4)
  w_ifco = array_ops.concat([w_i, w_f, w_c, w_o], axis=1)
  b_ifco = array_ops.concat([b_i, b_f, b_c, b_o], axis=0)
  return w_ifco, b_ifco


if __name__ == "__main__":
  test.main()
