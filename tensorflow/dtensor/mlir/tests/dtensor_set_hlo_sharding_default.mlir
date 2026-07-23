// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
// RUN: dtensor-opt %s -split-input-file -dtensor-set-hlo-sharding | FileCheck %s

// Check all inputs and operations have sharding attributes, with `check_layout_use_xla_spmd` set to default value (false).
// CHECK-LABEL: func @check_layouts_are_converted_to_xla_sharding_attributes
// CHECK-SAME: (%arg0: tensor<8x8xi32> {mhlo.sharding = "", tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"}) -> (tensor<8x8xi32> {mhlo.sharding = ""}) {
func.func @check_layouts_are_converted_to_xla_sharding_attributes(
  %arg0: tensor<8x8xi32> {tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"}) -> tensor<8x8xi32> {
  // CHECK:      "tf.DTensorLayout"
  // CHECK:      "tf.Identity"
  // CHECK:      "tf.DTensorLayout"
  // CHECK-NEXT: return
  %1 = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<8x8>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0>} : (tensor<8x8xi32>) -> tensor<8x8xi32>
  %2 = "tf.Identity"(%1) {_global_shape = [#tf_type.shape<8x8>], device = ""} : (tensor<8x8xi32>) -> tensor<8x8xi32>
  %3 = "tf.DTensorLayout"(%2) {global_shape = #tf_type.shape<8x8>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0>} : (tensor<8x8xi32>) -> tensor<8x8xi32>
  return %3 : tensor<8x8xi32>
}
