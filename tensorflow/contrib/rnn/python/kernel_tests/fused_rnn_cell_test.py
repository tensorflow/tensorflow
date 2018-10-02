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
"""Tests for tensorflow.contrib.rnn.python.ops.fused_rnn_cell."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.rnn.python.ops import fused_rnn_cell
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class FusedRnnCellTest(test.TestCase):

  def testBasicRNNFusedWrapper(self):
    """This test checks that using a wrapper for BasicRNN works as expected."""

    with self.cached_session() as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=19890212)
      cell = rnn_cell.BasicRNNCell(10)
      batch_size = 5
      input_size = 20
      timelen = 15
      inputs = constant_op.constant(
          np.random.randn(timelen, batch_size, input_size))
      with variable_scope.variable_scope("basic", initializer=initializer):
        unpacked_inputs = array_ops.unstack(inputs)
        outputs, state = rnn.static_rnn(
            cell, unpacked_inputs, dtype=dtypes.float64)
        packed_outputs = array_ops.stack(outputs)
        basic_vars = [
            v for v in variables.trainable_variables()
            if v.name.startswith("basic/")
        ]
        sess.run([variables.global_variables_initializer()])
        basic_outputs, basic_state = sess.run([packed_outputs, state])
        basic_grads = sess.run(gradients_impl.gradients(packed_outputs, inputs))
        basic_wgrads = sess.run(
            gradients_impl.gradients(packed_outputs, basic_vars))

      with variable_scope.variable_scope(
          "fused_static", initializer=initializer):
        fused_cell = fused_rnn_cell.FusedRNNCellAdaptor(
            rnn_cell.BasicRNNCell(10))
        outputs, state = fused_cell(inputs, dtype=dtypes.float64)
        fused_static_vars = [
            v for v in variables.trainable_variables()
            if v.name.startswith("fused_static/")
        ]
        sess.run([variables.global_variables_initializer()])
        fused_static_outputs, fused_static_state = sess.run([outputs, state])
        fused_static_grads = sess.run(gradients_impl.gradients(outputs, inputs))
        fused_static_wgrads = sess.run(
            gradients_impl.gradients(outputs, fused_static_vars))

      self.assertAllClose(basic_outputs, fused_static_outputs)
      self.assertAllClose(basic_state, fused_static_state)
      self.assertAllClose(basic_grads, fused_static_grads)
      for basic, fused in zip(basic_wgrads, fused_static_wgrads):
        self.assertAllClose(basic, fused, rtol=1e-2, atol=1e-2)

      with variable_scope.variable_scope(
          "fused_dynamic", initializer=initializer):
        fused_cell = fused_rnn_cell.FusedRNNCellAdaptor(
            rnn_cell.BasicRNNCell(10), use_dynamic_rnn=True)
        outputs, state = fused_cell(inputs, dtype=dtypes.float64)
        fused_dynamic_vars = [
            v for v in variables.trainable_variables()
            if v.name.startswith("fused_dynamic/")
        ]
        sess.run([variables.global_variables_initializer()])
        fused_dynamic_outputs, fused_dynamic_state = sess.run([outputs, state])
        fused_dynamic_grads = sess.run(
            gradients_impl.gradients(outputs, inputs))
        fused_dynamic_wgrads = sess.run(
            gradients_impl.gradients(outputs, fused_dynamic_vars))

      self.assertAllClose(basic_outputs, fused_dynamic_outputs)
      self.assertAllClose(basic_state, fused_dynamic_state)
      self.assertAllClose(basic_grads, fused_dynamic_grads)
      for basic, fused in zip(basic_wgrads, fused_dynamic_wgrads):
        self.assertAllClose(basic, fused, rtol=1e-2, atol=1e-2)

  def testTimeReversedFusedRNN(self):
    with self.cached_session() as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=19890213)
      fw_cell = rnn_cell.BasicRNNCell(10)
      bw_cell = rnn_cell.BasicRNNCell(10)
      batch_size = 5
      input_size = 20
      timelen = 15
      inputs = constant_op.constant(
          np.random.randn(timelen, batch_size, input_size))

      # test bi-directional rnn
      with variable_scope.variable_scope("basic", initializer=initializer):
        unpacked_inputs = array_ops.unstack(inputs)
        outputs, fw_state, bw_state = rnn.static_bidirectional_rnn(
            fw_cell, bw_cell, unpacked_inputs, dtype=dtypes.float64)
        packed_outputs = array_ops.stack(outputs)
        basic_vars = [
            v for v in variables.trainable_variables()
            if v.name.startswith("basic/")
        ]
        sess.run([variables.global_variables_initializer()])
        basic_outputs, basic_fw_state, basic_bw_state = sess.run(
            [packed_outputs, fw_state, bw_state])
        basic_grads = sess.run(gradients_impl.gradients(packed_outputs, inputs))
        basic_wgrads = sess.run(
            gradients_impl.gradients(packed_outputs, basic_vars))

      with variable_scope.variable_scope("fused", initializer=initializer):
        fused_cell = fused_rnn_cell.FusedRNNCellAdaptor(
            rnn_cell.BasicRNNCell(10))
        fused_bw_cell = fused_rnn_cell.TimeReversedFusedRNN(
            fused_rnn_cell.FusedRNNCellAdaptor(rnn_cell.BasicRNNCell(10)))
        fw_outputs, fw_state = fused_cell(
            inputs, dtype=dtypes.float64, scope="fw")
        bw_outputs, bw_state = fused_bw_cell(
            inputs, dtype=dtypes.float64, scope="bw")
        outputs = array_ops.concat([fw_outputs, bw_outputs], 2)
        fused_vars = [
            v for v in variables.trainable_variables()
            if v.name.startswith("fused/")
        ]
        sess.run([variables.global_variables_initializer()])
        fused_outputs, fused_fw_state, fused_bw_state = sess.run(
            [outputs, fw_state, bw_state])
        fused_grads = sess.run(gradients_impl.gradients(outputs, inputs))
        fused_wgrads = sess.run(gradients_impl.gradients(outputs, fused_vars))

      self.assertAllClose(basic_outputs, fused_outputs)
      self.assertAllClose(basic_fw_state, fused_fw_state)
      self.assertAllClose(basic_bw_state, fused_bw_state)
      self.assertAllClose(basic_grads, fused_grads)
      for basic, fused in zip(basic_wgrads, fused_wgrads):
        self.assertAllClose(basic, fused, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
  test.main()
