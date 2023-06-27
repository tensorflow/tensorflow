// RUN: dtensor-opt --nouse_layout_propagation_v2 -- %s -split-input-file -dtensor-annotate-global-shape -dtensor-layout-propagation -dtensor-spmd-expansion | FileCheck %s

// Check SPMD expansion of reduction op with replicated input.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>) {
  // CHECK:    "tf_device.cluster"
  // CHECK:      "tf.Sum"
  // CHECK-NOT:  "tf.DTensorAllReduce"
  // CHECK-NEXT: tf_device.return
  %0 = "tf_device.cluster"() ({
    %value = "tf.Const"() {value = dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<2x10xi32>,
                           _layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU"]}: () -> tensor<2x10xi32>
    %dimension = "tf.Const"() { value = dense<1> : tensor<i64>, _layout = ["sharding_specs:scalar, mesh:|x=2,y=2|*CPU"]} : () -> tensor<i64>
    %sum = "tf.Sum"(%value, %dimension) {keep_dims=true}: (tensor<2x10xi32>, tensor<i64>) -> tensor<2x1xi32>
    tf_device.return %sum : tensor<2x1xi32>
  }) {_mesh = "|x=2,y=2|*CPU"} : () -> (tensor<i32>)
  func.return
}

// -----

// Check SPMD expansion of reduction op on TPU mesh with replicated input.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>) {
  // CHECK:    "tf_device.cluster"
  // CHECK:      "tf.Sum"
  // CHECK-NOT:  tf.DTensorAllReduce
  // CHECK-NEXT: tf_device.return
  %0 = "tf_device.cluster"() ({
    %value = "tf.Const"() {value = dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<2x10xi32>,
                           _layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*TPU"]}: () -> tensor<2x10xi32>
    %dimension = "tf.Const"() { value = dense<1> : tensor<i64>, _layout = ["sharding_specs:scalar, mesh:|x=2,y=2|*TPU"] } : () -> tensor<i64>
    %sum = "tf.Sum"(%value, %dimension) {keep_dims=true}: (tensor<2x10xi32>, tensor<i64>) -> tensor<2x1xi32>
    tf_device.return %sum : tensor<2x1xi32>
  }) {_mesh = ["|x=2,y=2|*TPU"]} : () -> (tensor<i32>)
  func.return
}

// -----

// Check SPMD expansion of reduce op with sharded inputs on TPU mesh.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>) {
  // CHECK:    "tf_device.cluster"
  // CHECK:     tf.DTensorAllReduce
  %0 = "tf_device.cluster"() ({
    %value = "tf.Const"() {value = dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<2x10xi32>,
                           _layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*TPU"]}: () -> tensor<2x10xi32>
    %dimension = "tf.Const"() { value = dense<1> : tensor<i64>, _layout = ["sharding_specs:scalar, mesh:|x=2,y=2|*TPU"] } : () -> tensor<i64>
    %sum = "tf.Sum"(%value, %dimension) {keep_dims=true}: (tensor<2x10xi32>, tensor<i64>) -> tensor<2x1xi32>
    tf_device.return %sum : tensor<2x1xi32>
  }) {_mesh = "|x=2,y=2|*TPU"} : () -> (tensor<i32>)
  func.return
}

// -----

// Check SPMD reduction of reduce op with sharded inputs.
// CHECK-LABEL: func @main
// CHECK:  "tf_device.cluster"
// CHECK:  "tf.DTensorAllReduce"
func.func @main(%arg0: tensor<i32>) {
  %0 = "tf_device.cluster"() ({
    %value = "tf.Const"() {value = dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<2x10xi32>,
                           _layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*CPU"]}: () -> tensor<2x10xi32>
    %dimension = "tf.Const"() { value = dense<1> : tensor<i64>, _layout = ["sharding_specs:scalar, mesh:|x=2,y=2|*TPU"]} : () -> tensor<i64>
    %sum = "tf.Sum"(%value, %dimension) {keep_dims=true}: (tensor<2x10xi32>, tensor<i64>) -> tensor<2x1xi32>
    tf_device.return %sum : tensor<2x1xi32>
  }) {_mesh = "|x=2,y=2|*CPU"} : () -> (tensor<2x1xi32>)
  func.return
}

// -----

// Check that reduction over an unsharded dimension, should not emit an
// all-reduce.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>) {
  // CHECK:     "tf_device.cluster"
  // CHECK-NOT:   tf.DTensorAllReduce
  // CHECK:       tf_device.return
  %0 = "tf_device.cluster"() ({
    %value = "tf.Const"() {value = dense<[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]> : tensor<2x2x2xi32>,
                           _layout = ["sharding_specs:x,y,unsharded, mesh:|x=2,y=2|*TPU"]}: () -> tensor<2x2x2xi32>
    %dimension = "tf.Const"() { value = dense<2> : tensor<i64>, _layout = ["sharding_specs:scalar, mesh:|x=2,y=2|*TPU"] } : () -> tensor<i64>
    %sum = "tf.Sum"(%value, %dimension) {keep_dims=true}: (tensor<2x2x2xi32>, tensor<i64>) -> tensor<2x2x1xi32>
    tf_device.return %sum : tensor<2x2x1xi32>
  }) {_mesh = "|x=2,y=2|*TPU"} : () -> (tensor<i32>)
  func.return
}

