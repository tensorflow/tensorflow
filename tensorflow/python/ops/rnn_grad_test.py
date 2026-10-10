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

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_rnn_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


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

  def testBlockLSTMSeqLenMaxTooLarge(self):
    w, b, x, cs_prev, h_prev, w_peephole = self._block_lstm_inputs()
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError, r"seq_len_max must be between 0 and"
    ):
      self.evaluate(
          self._block_lstm(w, b, x, cs_prev, h_prev, w_peephole, seq_len_max=10)
      )

  def testBlockLSTMSeqLenMaxNegative(self):
    w, b, x, cs_prev, h_prev, w_peephole = self._block_lstm_inputs()
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError, r"seq_len_max must be between 0 and"
    ):
      self.evaluate(
          self._block_lstm(w, b, x, cs_prev, h_prev, w_peephole, seq_len_max=-1)
      )

  def testBlockLSTMGradSeqLenMaxTooLarge(self):
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError, r"seq_len_max must be between 0 and"
    ):
      self.evaluate(self._block_lstm_grad(seq_len_max=10))

  def testBlockLSTMGradSeqLenMaxNegative(self):
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError, r"seq_len_max must be between 0 and"
    ):
      self.evaluate(self._block_lstm_grad(seq_len_max=-1))

  def _block_lstm_inputs(
      self, num_steps=2, batch_size=1, input_size=1, hidden_size=4
  ):
    """Returns a valid BlockLSTM input set whose x has num_steps timesteps."""
    w = deterministic_random_uniform(
        [input_size + hidden_size, 4 * hidden_size]
    )
    b = deterministic_random_uniform([4 * hidden_size])
    x = deterministic_random_uniform([num_steps, batch_size, input_size])
    cs_prev = h_prev = deterministic_random_uniform([batch_size, hidden_size])
    w_peephole = array_ops.zeros(cs_prev.shape[1:], dtype=w.dtype)
    return w, b, x, cs_prev, h_prev, w_peephole

  def _block_lstm(self, w, b, x, cs_prev, h_prev, w_peephole, seq_len_max):
    return gen_rnn_ops.BlockLSTM(
        seq_len_max=math_ops.cast(seq_len_max, dtypes.int64),
        x=x,
        cs_prev=cs_prev,
        h_prev=h_prev,
        w=w,
        wci=w_peephole,
        wcf=w_peephole,
        wco=w_peephole,
        b=b,
        use_peephole=False,
    )

  def _block_lstm_grad(self, seq_len_max):
    """Runs a valid forward pass, then BlockLSTMGrad with seq_len_max."""
    w, b, x, cs_prev, h_prev, w_peephole = self._block_lstm_inputs()
    i, cs, f, o, ci, co, h = self._block_lstm(
        w, b, x, cs_prev, h_prev, w_peephole, seq_len_max=x.shape[0]
    )
    return gen_rnn_ops.BlockLSTMGrad(
        seq_len_max=math_ops.cast(seq_len_max, dtypes.int64),
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
        h=h,
        cs_grad=deterministic_random_uniform(cs.shape.as_list()),
        h_grad=deterministic_random_uniform(h.shape.as_list()),
        use_peephole=False,
    )

  def _run_with_unknown_shape(self, op, name, **kwargs):
    """Runs `op` with `kwargs[name]` fed through an unknown-shape signature.

    Hiding the shape keeps the op's shape function from rejecting the input
    while the graph is built, so the check under test is the kernel's.
    """
    value = kwargs.pop(name)
    if not context.executing_eagerly():
      # In graph mode the function call is inlined and shape inference sees
      # the argument's static shape, so hide it behind a placeholder as well.
      value = array_ops.placeholder_with_default(value, shape=None)

    @def_function.function(
        autograph=False,
        input_signature=[tensor_spec.TensorSpec(shape=None, dtype=value.dtype)],
    )
    def run(arg):
      return op(**{name: arg}, **kwargs)

    return self.evaluate(run(value))

  def _lstm_block_cell_inputs(self, batch_size=2, input_size=2, cell_size=2):
    cs_prev = deterministic_random_uniform([batch_size, cell_size])
    return dict(
        x=deterministic_random_uniform([batch_size, input_size]),
        cs_prev=cs_prev,
        h_prev=cs_prev,
        w=deterministic_random_uniform([input_size + cell_size, 4 * cell_size]),
        wci=deterministic_random_uniform([cell_size]),
        wcf=deterministic_random_uniform([cell_size]),
        wco=deterministic_random_uniform([cell_size]),
        b=deterministic_random_uniform([4 * cell_size]),
    )

  def _lstm_block_cell_grad_inputs(
      self, batch_size=2, input_size=2, cell_size=2
  ):
    kwargs = self._lstm_block_cell_inputs(batch_size, input_size, cell_size)
    for name in ("i", "cs", "f", "o", "ci", "co", "cs_grad", "h_grad"):
      kwargs[name] = deterministic_random_uniform([batch_size, cell_size])
    return kwargs

  def testLSTMBlockCellInvalidRank(self):
    # Test case for GitHub issue 113069. A rank 1 x was indexed at dimension 1
    # before the kernel's rank checks ran, which aborted the process.
    kwargs = self._lstm_block_cell_inputs()
    kwargs["x"] = deterministic_random_uniform([2])
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError, "x must be rank 2"
    ):
      self._run_with_unknown_shape(
          gen_rnn_ops.lstm_block_cell, "x", use_peephole=False, **kwargs
      )

  def testLSTMBlockCellInvalidPeepholeRank(self):
    # The peephole weights reach the functor as vec<T>(), a fatal check unless
    # the tensor is rank 1.
    kwargs = self._lstm_block_cell_inputs()
    kwargs["wci"] = deterministic_random_uniform([1, 2])
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError, "wci must be rank 1"
    ):
      self._run_with_unknown_shape(
          gen_rnn_ops.lstm_block_cell, "wci", use_peephole=True, **kwargs
      )

  def testLSTMBlockCellGradInvalidPeepholeRank(self):
    # As in the forward op, vec<T>() is called on the peephole weights even
    # when use_peephole is false.
    kwargs = self._lstm_block_cell_grad_inputs()
    kwargs["wci"] = deterministic_random_uniform([1, 2])
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError, "wci must be rank 1"
    ):
      self._run_with_unknown_shape(
          gen_rnn_ops.lstm_block_cell_grad, "wci", use_peephole=False, **kwargs
      )

  def testLSTMBlockCellInvalidPeepholeSize(self):
    # A peephole weight whose length differs from cell_size is broadcast
    # against [batch_size, cell_size] tensors and read out of bounds.
    for name in ["wci", "wcf", "wco"]:
      kwargs = self._lstm_block_cell_inputs()
      kwargs[name] = deterministic_random_uniform([3])
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          f"{name}.dim_size\\(0\\) != cell_size",
      ):
        self._run_with_unknown_shape(
            gen_rnn_ops.lstm_block_cell, name, use_peephole=True, **kwargs
        )

  def testLSTMBlockCellGradInvalidPeepholeSize(self):
    # wci_grad is allocated with the shape of wci but written with cell_size
    # elements.
    for name in ["wci", "wcf", "wco"]:
      kwargs = self._lstm_block_cell_grad_inputs()
      kwargs[name] = deterministic_random_uniform([3])
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          f"{name}.dim_size\\(0\\) != cell_size",
      ):
        self._run_with_unknown_shape(
            gen_rnn_ops.lstm_block_cell_grad, name, use_peephole=False, **kwargs
        )

  def _block_lstm_kwargs(self):
    w, b, x, cs_prev, h_prev, w_peephole = self._block_lstm_inputs()
    return dict(
        seq_len_max=math_ops.cast(x.shape[0], dtypes.int64),
        x=x,
        cs_prev=cs_prev,
        h_prev=h_prev,
        w=w,
        wci=w_peephole,
        wcf=w_peephole,
        wco=w_peephole,
        b=b,
        use_peephole=False,
    )

  def _block_lstm_grad_kwargs(self):
    kwargs = self._block_lstm_kwargs()
    i, cs, f, o, ci, co, h = gen_rnn_ops.BlockLSTM(**kwargs)
    kwargs.update(
        i=i,
        cs=cs,
        f=f,
        o=o,
        ci=ci,
        co=co,
        h=h,
        cs_grad=deterministic_random_uniform(cs.shape.as_list()),
        h_grad=deterministic_random_uniform(h.shape.as_list()),
    )
    return kwargs

  def testBlockLSTMInvalidPeepholeSize(self):
    for name in ["wci", "wcf", "wco"]:
      kwargs = self._block_lstm_kwargs()
      kwargs[name] = deterministic_random_uniform([3])
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          f"{name}.dim_size\\(0\\) != cell_size",
      ):
        self._run_with_unknown_shape(gen_rnn_ops.BlockLSTM, name, **kwargs)

  def testBlockLSTMGradInvalidPeepholeSize(self):
    # As in LSTMBlockCellGrad, wci_grad is allocated with the shape of wci but
    # written with cell_size elements.
    for name in ["wci", "wcf", "wco"]:
      kwargs = self._block_lstm_grad_kwargs()
      kwargs[name] = deterministic_random_uniform([3])
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          f"{name}.dim_size\\(0\\) != cell_size",
      ):
        self._run_with_unknown_shape(gen_rnn_ops.BlockLSTMGrad, name, **kwargs)

  def testBlockLSTMGradInvalidBiasSize(self):
    # A bias of length 4 * cell_size + 3 passed the old
    # `cell_size == b.dim_size(0) / 4` check through integer division.
    kwargs = self._block_lstm_grad_kwargs()
    cell_size = kwargs["cs_prev"].shape[1]
    kwargs["b"] = deterministic_random_uniform([4 * cell_size + 3])
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError, "w and b cell_size don't match"
    ):
      self._run_with_unknown_shape(gen_rnn_ops.BlockLSTMGrad, "b", **kwargs)

  def testLSTMBlockCellGradEmptyCsPrev(self):
    # The forward op rejects an empty cs_prev (#58270); the gradient must
    # reject the same inputs.
    kwargs = self._lstm_block_cell_grad_inputs(cell_size=0)
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError, "cs_prev_tensor is empty"
    ):
      self._run_with_unknown_shape(
          gen_rnn_ops.lstm_block_cell_grad,
          "cs_prev",
          use_peephole=False,
          **kwargs,
      )

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
