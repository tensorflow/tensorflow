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

# RUN: %p/shapes_for_arguments | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common


class TestModule(tf.Module):

  # Check that we get shapes annotated on function arguments.
  #
  # Besides checking the shape on the function input argument, this test also
  # checks that the shape on the input argument is propagated to the return
  # value.
  # We eventually want to move the shape inference to a pass separate from
  # the initial import, in which case that aspect of this test doesn't make much
  # sense and will be superceded by MLIR->MLIR shape inference tests.
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}(%arg0: tensor<f32> {{.*}}) -> (tensor<f32> {{.*}})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["some_function"]
  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def some_function(self, x):
    return x


if __name__ == '__main__':
  common.do_test(TestModule)
