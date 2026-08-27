# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for quantized Dense layer."""

import numpy as np
import tensorflow as tf

from tensorflow.python.keras.layers.quantized import QuantizedDense
from tensorflow.python.platform import test


class QuantizedDenseTest(test.TestCase):

    def setUp(self):
        super(QuantizedDenseTest, self).setUp()
        tf.random.set_seed(0)
        np.random.seed(0)

    def test_quantized_dense_basic(self):
        inputs = tf.random.uniform((32, 128))

        # Test 8-bit quantization
        layer_8bit = QuantizedDense(64, bits=8)
        out_8bit = layer_8bit(inputs)

        self.assertEqual(out_8bit.shape, (32, 64))
        self.assertEqual(layer_8bit.kernel.shape, (128, 64))
        self.assertEqual(layer_8bit.bias.shape, (64,))

    def test_quantized_dense_4bit(self):
        inputs = tf.random.uniform((16, 32))

        # Test 4-bit quantization
        layer_4bit = QuantizedDense(16, bits=4, use_bias=False)
        out_4bit = layer_4bit(inputs)

        self.assertEqual(out_4bit.shape, (16, 16))
        self.assertIsNone(layer_4bit.bias)

    def test_invalid_bits(self):
        with self.assertRaisesRegex(ValueError,
                                    "Only 4-bit and 8-bit quantization"):
            QuantizedDense(32, bits=16)

if __name__ == "__main__":
    test.main()
