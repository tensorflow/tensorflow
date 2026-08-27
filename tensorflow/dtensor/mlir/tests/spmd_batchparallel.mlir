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
// RUN: dtensor-opt %s -split-input-file -dtensor-annotate-global-shape -dtensor-spmd-expansion -verify-diagnostics | FileCheck %s --dump-input=fail

// Test replicated layout.

func.func @main(%arg0: tensor<1xf32>,
           %arg1: tensor<8x128x128x3xf32> {tf._layout = "sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:TPU|x=2,y=2|*TPU"}) -> tensor<8x128x128x3xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT: "tf.AdjustSaturation"
  // CHECK-NEXT:        tf_device.return
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<1>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|*TPU>} : (tensor<1xf32>) -> tensor<1xf32>
    %2 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x128x128x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:TPU|x=2,y=2|*TPU>} : (tensor<8x128x128x3xf32>) -> tensor<8x128x128x3xf32>
    %3 = "tf.AdjustSaturation"(%2, %1) {} : (tensor<8x128x128x3xf32>, tensor<1xf32>) -> tensor<8x128x128x3xf32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<8x128x128x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded,unsharded, mesh:TPU|x=2,y=2|*TPU>} : (tensor<8x128x128x3xf32>) -> tensor<8x128x128x3xf32>
    tf_device.return %4 : tensor<8x128x128x3xf32>
  }) {_mesh = "TPU|x=2,y=2|*TPU"} : () -> tensor<8x128x128x3xf32>
  func.return %0 : tensor<8x128x128x3xf32>
}

// -----

// Test batch sharded layout. Should emit Identity op.

func.func @main(%arg0: tensor<1xf32>,
           %arg1: tensor<8x128x128x3xf32> {tf._layout = "sharding_specs:x,unsharded,unsharded,unsharded, mesh:TPU|x=2,y=2|*TPU"}) -> tensor<8x128x128x3xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT: "tf.AdjustSaturation"
  // CHECK-NEXT: "tf.IdentityN"
  // CHECK-NEXT:        tf_device.return
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<1>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|*TPU>} : (tensor<1xf32>) -> tensor<1xf32>
    %2 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x128x128x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:TPU|x=2,y=2|*TPU>} : (tensor<8x128x128x3xf32>) -> tensor<8x128x128x3xf32>
    %3 = "tf.AdjustSaturation"(%2, %1) {} : (tensor<8x128x128x3xf32>, tensor<1xf32>) -> tensor<8x128x128x3xf32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<8x128x128x3>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded,unsharded, mesh:TPU|x=2,y=2|*TPU>} : (tensor<8x128x128x3xf32>) -> tensor<8x128x128x3xf32>
    tf_device.return %4 : tensor<8x128x128x3xf32>
  }) {_mesh = "TPU|x=2,y=2|*TPU"} : () -> tensor<8x128x128x3xf32>
  func.return %0 : tensor<8x128x128x3xf32>
}
