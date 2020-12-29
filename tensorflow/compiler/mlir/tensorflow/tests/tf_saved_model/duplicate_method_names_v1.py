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

# RUN: %p/duplicate_method_names_v1 | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common_v1

# Tests different SignatureDef's with identical method_name string

# CHECK:      func {{@[a-zA-Z_0-9]+}}(
# CHECK-SAME:   {{.*}})
# CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["key"]

# CHECK:      func {{@[a-zA-Z_0-9]+}}(
# CHECK-SAME:   {{.*}})
# CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["key2"]


def Test():

  x = tf.constant(1.0, shape=(3, 3))
  y = tf.constant(1.0, shape=(3, 3))

  s = tf.transpose(x)
  t = tf.transpose(y)

  tensor_info_s = tf.compat.v1.saved_model.utils.build_tensor_info(s)
  tensor_info_t = tf.compat.v1.saved_model.utils.build_tensor_info(t)

  signature_def = tf.saved_model.signature_def_utils.build_signature_def(
      inputs=None, outputs={'s': tensor_info_s}, method_name='some_function')
  signature_def2 = tf.saved_model.signature_def_utils.build_signature_def(
      inputs=None, outputs={'t': tensor_info_t}, method_name='some_function')

  # Create two signatures that share the same variable.
  return {'key': signature_def, 'key2': signature_def2}, None, None


if __name__ == '__main__':
  common_v1.set_tf_options()
  common_v1.do_test(Test)
