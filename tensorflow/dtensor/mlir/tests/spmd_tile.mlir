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

