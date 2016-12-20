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

import tensorflow as tf


class FusedRnnCellTest(tf.test.TestCase):

  def testBasicRNNFusedWrapper(self):
    """This test checks that using a wrapper for BasicRNN works as expected."""

    with self.test_session() as sess:
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=19890212)
      cell = tf.contrib.rnn.BasicRNNCell(10)
      batch_size = 5
      input_size = 20
      timelen = 15
      inputs = tf.constant(np.random.randn(timelen, batch_size, input_size))
      with tf.variable_scope("basic", initializer=initializer):
        unpacked_inputs = tf.unstack(inputs)
        outputs, state = tf.contrib.rnn.static_rnn(
            cell, unpacked_inputs, dtype=tf.float64)
        packed_outputs = tf.stack(outputs)
        basic_vars = [v for v in tf.trainable_variables()
                      if v.name.startswith("basic/")]
        sess.run([tf.global_variables_initializer()])
        basic_outputs, basic_state = sess.run([packed_outputs, state])
        basic_grads = sess.run(tf.gradients(packed_outputs, inputs))
        basic_wgrads = sess.run(tf.gradients(packed_outputs, basic_vars))

      with tf.variable_scope("fused_static", initializer=initializer):
        fused_cell = tf.contrib.rnn.FusedRNNCellAdaptor(cell)
        outputs, state = fused_cell(inputs, dtype=tf.float64)
        fused_static_vars = [v for v in tf.trainable_variables()
                             if v.name.startswith("fused_static/")]
        sess.run([tf.global_variables_initializer()])
        fused_static_outputs, fused_static_state = sess.run([outputs, state])
        fused_static_grads = sess.run(tf.gradients(outputs, inputs))
        fused_static_wgrads = sess.run(tf.gradients(outputs, fused_static_vars))

      self.assertAllClose(basic_outputs, fused_static_outputs)
      self.assertAllClose(basic_state, fused_static_state)
      self.assertAllClose(basic_grads, fused_static_grads)
      for basic, fused in zip(basic_wgrads, fused_static_wgrads):
        self.assertAllClose(basic, fused, rtol=1e-2, atol=1e-2)

      with tf.variable_scope("fused_dynamic", initializer=initializer):
        fused_cell = tf.contrib.rnn.FusedRNNCellAdaptor(
            cell, use_dynamic_rnn=True)
        outputs, state = fused_cell(inputs, dtype=tf.float64)
        fused_dynamic_vars = [v for v in tf.trainable_variables()
                              if v.name.startswith("fused_dynamic/")]
        sess.run([tf.global_variables_initializer()])
        fused_dynamic_outputs, fused_dynamic_state = sess.run([outputs, state])
        fused_dynamic_grads = sess.run(tf.gradients(outputs, inputs))
        fused_dynamic_wgrads = sess.run(
            tf.gradients(outputs, fused_dynamic_vars))

      self.assertAllClose(basic_outputs, fused_dynamic_outputs)
      self.assertAllClose(basic_state, fused_dynamic_state)
      self.assertAllClose(basic_grads, fused_dynamic_grads)
      for basic, fused in zip(basic_wgrads, fused_dynamic_wgrads):
        self.assertAllClose(basic, fused, rtol=1e-2, atol=1e-2)

  def testTimeReversedFusedRNN(self):
    with self.test_session() as sess:
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=19890213)
      cell = tf.contrib.rnn.BasicRNNCell(10)
      batch_size = 5
      input_size = 20
      timelen = 15
      inputs = tf.constant(np.random.randn(timelen, batch_size, input_size))

      # test bi-directional rnn
      with tf.variable_scope("basic", initializer=initializer):
        unpacked_inputs = tf.unstack(inputs)
        outputs, fw_state, bw_state = tf.contrib.rnn.static_bidirectional_rnn(
            cell, cell, unpacked_inputs, dtype=tf.float64)
        packed_outputs = tf.stack(outputs)
        basic_vars = [v for v in tf.trainable_variables()
                      if v.name.startswith("basic/")]
        sess.run([tf.global_variables_initializer()])
        basic_outputs, basic_fw_state, basic_bw_state = sess.run(
            [packed_outputs, fw_state, bw_state])
        basic_grads = sess.run(tf.gradients(packed_outputs, inputs))
        basic_wgrads = sess.run(tf.gradients(packed_outputs, basic_vars))

      with tf.variable_scope("fused", initializer=initializer):
        fused_cell = tf.contrib.rnn.FusedRNNCellAdaptor(cell)
        fused_bw_cell = tf.contrib.rnn.TimeReversedFusedRNN(fused_cell)
        fw_outputs, fw_state = fused_cell(inputs, dtype=tf.float64, scope="fw")
        bw_outputs, bw_state = fused_bw_cell(
            inputs, dtype=tf.float64, scope="bw")
        outputs = tf.concat_v2([fw_outputs, bw_outputs], 2)
        fused_vars = [v for v in tf.trainable_variables()
                      if v.name.startswith("fused/")]
        sess.run([tf.global_variables_initializer()])
        fused_outputs, fused_fw_state, fused_bw_state = sess.run(
            [outputs, fw_state, bw_state])
        fused_grads = sess.run(tf.gradients(outputs, inputs))
        fused_wgrads = sess.run(tf.gradients(outputs, fused_vars))

      self.assertAllClose(basic_outputs, fused_outputs)
      self.assertAllClose(basic_fw_state, fused_fw_state)
      self.assertAllClose(basic_bw_state, fused_bw_state)
      self.assertAllClose(basic_grads, fused_grads)
      for basic, fused in zip(basic_wgrads, fused_wgrads):
        self.assertAllClose(basic, fused, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
  tf.test.main()
