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

# RUN: %p/no_input_shape_v1 | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
import tensorflow.compat.v1 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common_v1
from tensorflow.core.protobuf import meta_graph_pb2

# Verify that the tf.versions attribute exists. It is difficult to enforce
# contents, since the version numbers change over time. The conversion logic
# itself is verified in the common graphdef converter, so here just assert
# it is being invoked.
# CHECK: module
# CHECK-SAME: tf.versions
# CHECK-SAME: bad_consumers
# CHECK-SAME: min_consumer
# CHECK-SAME: producer

# CHECK:      func {{@[a-zA-Z_0-9]+}}(
# CHECK-SAME:   [[ARG:%.*]]: tensor<*xf32> {tf_saved_model.index_path = ["x"]}

# CHECK: [[shape:%.*]] = "tf.Shape"([[ARG]])
# CHECK-NEXT: [[batch_size:%.*]] = "tf.StridedSlice"([[shape]],
# CHECK-NEXT: [[result:%.*]] = "tf.Pack"([[batch_size]],
# CHECK-NEXT: return [[result]] : tensor<2xi32>


def Test():

  x = tf.placeholder(dtype=tf.float32, shape=[None])
  batch_size = tf.shape(x)[0]
  r = tf.convert_to_tensor([batch_size, 1])

  tensor_info_x = meta_graph_pb2.TensorInfo(
      name=x.name, dtype=tf.as_dtype(x.dtype).as_datatype_enum)
  tensor_info_r = tf.compat.v1.saved_model.utils.build_tensor_info(r)

  return {
      'key': (tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
          inputs={'x': tensor_info_x},
          outputs={'r': tensor_info_r},
          method_name='some_function'))
  }, None, None


if __name__ == '__main__':
  common_v1.set_tf_options()
  common_v1.do_test(Test)
