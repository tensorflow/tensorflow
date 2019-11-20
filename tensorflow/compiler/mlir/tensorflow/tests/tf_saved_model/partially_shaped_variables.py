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

# RUN: %p/partially_shaped_variables | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common


class TestModule(tf.Module):

  def __init__(self):
    super(TestModule, self).__init__()
    # CHECK: "tf_saved_model.global_tensor"() {is_mutable, {{.*}} tf_saved_model.exported_names = ["v0"], type = tensor<*xf32>, value = dense<0.000000e+00> : tensor<1xf32>} : () -> ()
    # CHECK: "tf_saved_model.global_tensor"() {is_mutable, {{.*}} tf_saved_model.exported_names = ["v1"], type = tensor<?xf32>, value = dense<0.000000e+00> : tensor<1xf32>} : () -> ()
    self.v0 = tf.Variable([0.], shape=tf.TensorShape(None))
    self.v1 = tf.Variable([0.], shape=[None])


if __name__ == '__main__':
  common.do_test(TestModule, exported_names=[])
