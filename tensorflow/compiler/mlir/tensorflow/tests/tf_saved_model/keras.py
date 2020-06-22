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

# RUN: %p/keras | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common


def mnist_model():
  """Creates a MNIST model."""
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(128, activation='relu'))
  model.add(tf.keras.layers.Dense(10, activation='softmax'))
  return model


class TestModule(tf.Module):

  def __init__(self):
    super(TestModule, self).__init__()
    self.model = mnist_model()

  # CHECK: func {{@[a-zA-Z_0-9]+}}(%arg0: tensor<1x28x28x1xf32> {tf._user_specified_name = "x", tf_saved_model.index_path = [0]}
  # CHECK: attributes {{.*}} tf_saved_model.exported_names = ["my_predict"]
  @tf.function(input_signature=[
      tf.TensorSpec([1, 28, 28, 1], tf.float32),
  ])
  def my_predict(self, x):
    return self.model(x)


if __name__ == '__main__':
  common.do_test(TestModule, exported_names=['my_predict'])
