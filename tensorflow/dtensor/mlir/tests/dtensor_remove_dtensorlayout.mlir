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
// RUN: dtensor-opt %s -split-input-file -dtensor-remove-dtensorlayout | FileCheck %s

// This test checks DTensorLayout ops are all removed, regardless of whether it
// has the `use_xla_spmd` attribute.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>) {
  %0 = "tf_device.cluster"() ({
    // CHECK:      "tf.Const"()
    // CHECK-NOT:  "tf.DTensorLayout"
    // CHECK:      "tf.Const"()
    // CHECK-NOT:  "tf.DTensorLayout"
    // CHECK:      "tf.Add"
    // CHECK-NOT:  "tf.DTensorLayout"
    // CHECK-NEXT: tf_device.return
    %1 = "tf.Const"() {value = dense<[[4, 1], [2, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %2 = "tf.DTensorLayout"(%1) {global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3|use_xla_spmd>} : (tensor<2x2xi32>) -> (tensor<2x2xi32>)
    %3 = "tf.Const"() {value = dense<[[1, 1], [2, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %4 = "tf.DTensorLayout"(%2) {global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2x2xi32>) -> (tensor<2x2xi32>)
    %5 = "tf.Add"(%2, %4): (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    %6 = "tf.DTensorLayout"(%5) {global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2x2xi32>) -> (tensor<2x2xi32>)
    tf_device.return %6 : tensor<2x2xi32>
  }) {_mesh = "mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> (tensor<2x2xi32>)
  func.return
}