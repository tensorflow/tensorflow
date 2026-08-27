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
// RUN: dtensor-opt %s -split-input-file -dtensor-annotate-global-shape -dtensor-layout-propagation -dtensor-spmd-expansion -verify-diagnostics | FileCheck %s

// Check TileOp on sharded const input.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>,
  %arg1: tensor<2x1xf32> {tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU"},
  %arg2: tensor<2xi32> {tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=2|*CPU"}) -> (tensor<4x3xf32>{
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[MULTIPLES:.*]] = "tf.Const"() {value = dense<[1, 3]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK:      "tf.Tile"(%arg1, %[[MULTIPLES]])
  // CHECK-SAME:  (tensor<2x1xf32>, tensor<2xi32>) -> tensor<2x3xf32>
  tf._default_layout = "sharding_specs:x,unsharded, mesh:|x=2,y=2|*CPU"}) {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const"() {value = dense<[2, 3]> : tensor<2xi32>, _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|*CPU"]} : () -> tensor<2xi32>
    %2 = "tf.Tile"(%arg1, %1) {device = ""} : (tensor<2x1xf32>, tensor<2xi32>) -> tensor<4x3xf32>
    tf_device.return %2 : tensor<4x3xf32>
  }) {} : () -> tensor<4x3xf32>
  func.return %0 : tensor<4x3xf32>
}

// -----

// Check TileOp on sharded const input with partial shape.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>,
  %arg1: tensor<?x1xf32> {tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU"},
  %arg2: tensor<2xi32> {tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=2|*CPU"}) -> (tensor<?x3xf32>{
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[MULTIPLES:.*]] = "tf.Const"()
  // CHECK-NEXT: dense<[1, 3]>
  // CHECK:      "tf.Tile"
  // CHECK-SAME: (tensor<?x1xf32>, tensor<2xi32>) -> tensor<?x3xf32>
  tf._default_layout = "sharding_specs:x,unsharded, mesh:|x=2,y=2|*CPU"}) {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const"() {value = dense<[2, 3]> : tensor<2xi32>, _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|*CPU"]} : () -> tensor<2xi32>
    %2 = "tf.Tile"(%arg1, %1) {device = ""} : (tensor<?x1xf32>, tensor<2xi32>) -> tensor<?x3xf32>
    tf_device.return %2 : tensor<?x3xf32>
  }) {} : () -> tensor<?x3xf32>
  func.return %0 : tensor<?x3xf32>
}

// -----

// Check TileOp on sharded const input with partial shape.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>,
  %arg1: tensor<?x1xf32> {tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU"},
  %arg2: tensor<2xi32> {tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=2|*CPU"}) -> (tensor<?x3xf32>{
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[MULTIPLES:.*]] = "tf.Const"()
  // CHECK-NEXT: dense<[1, 3]>
  // CHECK:      "tf.Tile"
  // CHECK-SAME: (tensor<?x1xf32>, tensor<2xi32>) -> tensor<?x3xf32>
  tf._default_layout = "sharding_specs:x,unsharded, mesh:|x=2,y=2|*CPU"}) {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const"() {value = dense<[2, 3]> : tensor<2xi32>, _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|*CPU"]} : () -> tensor<2xi32>
    %2 = "tf.Tile"(%arg1, %1) {device = ""} : (tensor<?x1xf32>, tensor<2xi32>) -> tensor<?x3xf32>
    tf_device.return %2 : tensor<?x3xf32>
  }) {} : () -> tensor<?x3xf32>
  func.return %0 : tensor<?x3xf32>
}

