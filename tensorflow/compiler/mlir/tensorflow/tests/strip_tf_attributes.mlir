// Copyright 2026 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: tf-opt %s -allow-unregistered-dialect --tf-strip-tf-attributes --split-input-file | FileCheck %s

// CHECK-LABEL: strips_attributes
// CHECK-NOT: tf
func.func @strips_attributes(%arg0: tensor<32x28x28x1xf32> {tf._user_specified_name = "x"},
                             %arg1: tensor<3x3x1x5xf32> {tf._user_specified_name = "w1"},
                             %arg2: tensor<5xf32> {tf._user_specified_name = "b1"},
                             %arg3: tensor<3920x10xf32> {tf._user_specified_name = "w2"},
                             %arg4: tensor<10xf32> {tf._user_specified_name = "b2"}) -> tensor<10xf32>
          attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<32x28x28x1>, #tf_type.shape<3x3x1x5>, #tf_type.shape<5>, #tf_type.shape<3920x10>, #tf_type.shape<10>]} {
  return %arg4 : tensor<10xf32>
}

// -----

// CHECK-LABEL: strips_attributes_with_tf_values
// CHECK-NOT: tf
func.func @strips_attributes_with_tf_values()
          attributes {foo = #tf_type.shape<32x28x28x1>} {
  return
}

// -----

// CHECK-LABEL: strips_result_attributes
// CHECK-NOT: tf
func.func @strips_result_attributes() -> (f32 {tf.foo = "bar"}) {
  %0 = "foo.constant"() : () -> f32
  return %0 : f32
}

// -----

// CHECK-LABEL: strips_module_attributes
module @strips_module_attributes attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1235 : i32}} {
// CHECK-NOT: tf
// CHECK: body
  func.func private @body() -> () {
    return
  }
}
