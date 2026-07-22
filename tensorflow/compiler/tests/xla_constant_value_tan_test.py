# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Regression tests for XLA constant-value inference of tf.math.tan.

Covers github.com/tensorflow/tensorflow/issues/122050:
  AnalyzeConstantValueFallback can't handle opcode: tan

The root cause was that kTan was absent from the AnalyzeConstant switch in
xla/hlo/builder/value_inference.cc, causing a fall-through to
AnalyzeConstantValueFallback which returned InvalidArgument for it. kTan is
now handled alongside kSin/kCos, delegating to HloProtoEvaluator.
"""

import tensorflow as tf


class TanConstantValueInferenceTest(tf.test.TestCase):
  """Tests that tf.math.tan works under jit_compile=True constant analysis."""

  def test_real_tan_jit_compile(self):
    """Real-valued tan feeding a shape-determining op compiles correctly."""

    @tf.function(jit_compile=True)
    def model(x):
      r = tf.math.tan(x)
      limit = tf.cast(tf.reduce_max(r), tf.int32)
      return tf.range(limit)

    x = tf.constant([0.1, 0.2, 0.3], dtype=tf.float32)
    eager_out = model.python_function(x)
    xla_out = model(x)
    self.assertEqual(eager_out.shape, xla_out.shape)
    self.assertAllEqual(eager_out, xla_out)

  def test_complex_tan_sequence_mask_jit_compile(self):
    """Exact repro from issue #122050: complex tan -> imag -> sequence_mask."""
    tf.random.set_seed(590)
    t = tf.complex(
        tf.random.normal([2, 8, 16, 16, 8], dtype=tf.float32),
        tf.random.normal([2, 8, 16, 16, 8], dtype=tf.float32),
    )

    @tf.function(jit_compile=True)
    def model(t):
      r = tf.math.tan(t)
      i = tf.experimental.numpy.imag(r)
      return tf.sequence_mask(i, dtype=tf.int32)

    eager_out = model.python_function(t)
    # Must not raise InvalidArgumentError about kTan.
    xla_out = model(t)
    self.assertEqual(eager_out.shape, xla_out.shape)
    self.assertAllEqual(eager_out, xla_out)

  def test_tan_sin_cos_consistency(self):
    """kSin and kCos were already handled; kTan must behave identically."""

    @tf.function(jit_compile=True)
    def model_sin(x):
      s = tf.math.sin(x)
      return tf.sequence_mask(s, dtype=tf.int32)

    @tf.function(jit_compile=True)
    def model_cos(x):
      c = tf.math.cos(x)
      return tf.sequence_mask(c, dtype=tf.int32)

    @tf.function(jit_compile=True)
    def model_tan(x):
      tt = tf.math.tan(x)
      return tf.sequence_mask(tt, dtype=tf.int32)

    x = tf.constant([[0.1, 0.5], [0.2, 0.9]], dtype=tf.float32)
    sin_out = model_sin(x)
    cos_out = model_cos(x)
    tan_out = model_tan(x)

    # All three transcendentals must produce results of the same rank.
    self.assertEqual(sin_out.shape.rank, cos_out.shape.rank)
    self.assertEqual(cos_out.shape.rank, tan_out.shape.rank)
    # All three transcendentals must produce correct results matching eager execution.
    self.assertAllEqual(sin_out, model_sin.python_function(x))
    self.assertAllEqual(cos_out, model_cos.python_function(x))
    self.assertAllEqual(tan_out, model_tan.python_function(x))


if __name__ == "__main__":
  tf.test.main()
