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
"""Transpose convolution layers validation test."""

import tensorflow.compat.v2 as tf

from tensorflow.python.keras.layers import Conv1DTranspose
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import Conv3DTranspose
from tensorflow.python.keras.testing_infra import test_combinations


@test_combinations.run_all_keras_modes
class ConvTransposeValidationTest(test_combinations.TestCase):
    def test_conv_transpose_invalid_parameters(self):
        for layer_cls in [Conv1DTranspose, Conv2DTranspose, Conv3DTranspose]:
            with self.assertRaisesRegex(
                ValueError,
                "if any value of `strides` is > 1, then all values of "
                "`dilation_rate` must be 1.",
            ):
                layer_cls(
                    filters=1, kernel_size=3, strides=2, dilation_rate=2
                )


if __name__ == "__main__":
    tf.test.main()
