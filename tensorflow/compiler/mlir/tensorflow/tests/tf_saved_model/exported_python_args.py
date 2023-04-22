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

# RUN: (! %p/exported_python_args 2>&1) | FileCheck %s

# pylint: disable=missing-docstring,line-too-long,dangerous-default-value
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common


class TestModule(tf.Module):

  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def some_function(self, x):
    return self.callee(x)

  # CHECK: While importing SavedModel function 'callee': in input signature:
  # CHECK-SAME: Unhandled structured value kind {{.*}} at index path: <value>.1.foo
  @tf.function
  def callee(self, x, n={'foo': 42}):
    return x


if __name__ == '__main__':
  common.do_test(TestModule)
