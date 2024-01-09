// RUN: dtensor-opt -- %s -split-input-file -dtensor-annotate-global-shape -dtensor-layout-propagation -dtensor-spmd-expansion | FileCheck %s

// SPMD of shape of with replicated layout.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>) {
  // CHECK:    "tf_device.cluster"
  // CHECK:      "tf.Shape"
  // CHECK-NOT:  "tf.Const"()
  // CHECK-NOT:  %[[C:.*]] = "tf.Mul"
  // CHECK-NEXT: tf_device.return
  %0 = "tf_device.cluster"() ({
    %a = "tf.Const"() {value = dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<2x10xi32>,
                       _layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU"]}: () -> tensor<2x10xi32>
    %shape_a = "tf.Shape"(%a) : (tensor<2x10xi32>) -> tensor<2xi32>
    tf_device.return %shape_a : tensor<2xi32>
  }) {_mesh = "|x=2,y=2|*CPU"} : () -> (tensor<i32>)
  func.return
}

// -----

// Check SPMD of shape op with 2D input.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>) {
  // CHECK:    "tf_device.cluster"
  // CHECK: "tf.Shape"
  // CHECK: "tf.Const"() {value = dense<[2, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: %[[C:.*]] = "tf.Mul"(%[[A:.*]], %[[B:.*]]) {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  // CHECK-NEXT: tf_device.return
  %0 = "tf_device.cluster"() ({
    %a = "tf.Const"() {value = dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]> : tensor<2x10xi32>,
                       _layout = ["sharding_specs:x,unsharded, mesh:|x=2,y=2|*CPU"]}: () -> tensor<2x10xi32>
    %shape_a = "tf.Shape"(%a) : (tensor<2x10xi32>) -> tensor<2xi32>
    tf_device.return %shape_a : tensor<2xi32>
  }) {_mesh = "|x=2,y=2|*CPU"} : () -> (tensor<i32>)
  func.return
}

// -----

// Check SPMD of shape op with 3D input.
// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]: tensor<i32>
func.func @main(%arg0: tensor<i32>) {
  // CHECK:    "tf_device.cluster"
  // CHECK-NEXT:  %[[MOD_CONST:[0-9]*]] = "tf.Const"
  // CHECK-NEXT:  %[[DIV_CONST:[0-9]*]] = "tf.Const"
  // CHECK-NEXT:  %[[PRE_MESH_COORDS:[0-9]*]] = "tf.Div"(%[[ARG0]], %[[DIV_CONST]])
  // CHECK-NEXT:  %[[MESH_COORDS:[0-9]*]] = "tf.FloorMod"(%[[PRE_MESH_COORDS]], %[[MOD_CONST]])
  // CHECK-NEXT:  %[[TENSOR:[0-9]*]] = "tf.Const"
  // CHECK-NEXT:  %[[SLICE_SHAPE:[0-9]*]] = "tf.Const"
  // CHECK-NEXT:  %[[PRE_SLICE_OFFSET:[0-9]*]] = "tf.Const"
  // CHECK-NEXT:  %[[SLICE_OFFSET:[0-9]*]] = "tf.MatMul"(%[[MESH_COORDS]], %[[PRE_SLICE_OFFSET]])
  // CHECK-NEXT:  %[[SQUEEZED_OFFSET:[0-9]*]] = "tf.Squeeze"(%[[SLICE_OFFSET]])
  // CHECK-NEXT:  %[[TENSOR_SLICE:[0-9]*]] = "tf.Slice"(%[[TENSOR]], %[[SQUEEZED_OFFSET]], %[[SLICE_SHAPE]])
  // CHECK-NEXT:  %[[TENSOR_SLICE_IDENTITY:[0-9]*]] = "tf.IdentityN"(%[[TENSOR_SLICE]])
  // CHECK-NEXT:  %[[TENSOR_SLICE_SHAPE:[0-9]*]] = "tf.Shape"(%[[TENSOR_SLICE_IDENTITY]])
  // CHECK-NEXT:  %[[TENSOR_SPLIT_SIZES:[0-9]*]] = "tf.Const"
  // CHECK-NEXT:  %[[TENSOR_SHAPE:[0-9]*]] = "tf.Mul"(%[[TENSOR_SLICE_SHAPE]], %[[TENSOR_SPLIT_SIZES]])
  // CHECK-NEXT:  tf_device.return
  %0 = "tf_device.cluster"() ({

    %1 = "tf.Const"() {value = dense<[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]> : tensor<2x2x2xi32>,
                       _layout = ["sharding_specs:x,y,unsharded, mesh:|x=2,y=2|*CPU"]}: () -> tensor<2x2x2xi32>
    %2 = "tf.Shape"(%1) : (tensor<2x2x2xi32>) -> tensor<3xi32>
    tf_device.return %2 : tensor<3xi32>
  }) {_mesh = "|x=2,y=2|*CPU"} : () -> (tensor<3xi32>)
  func.return
}

// -----

// Check SPMD of rank op with 3D input.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>) {
  // CHECK:    "tf_device.cluster"
  // CHECK: "tf.Rank"
  // CHECK-NEXT: tf_device.return
  %0 = "tf_device.cluster"() ({
    %b = "tf.Const"() {value = dense<[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]> : tensor<2x2x2xi32>,
                       _layout = ["sharding_specs:x,y,unsharded, mesh:|x=2,y=2|*CPU"]}: () -> tensor<2x2x2xi32>
    %rank_b = "tf.Rank"(%b) : (tensor<2x2x2xi32>) -> tensor<i32>
    tf_device.return %rank_b : tensor<i32>
  }) {_mesh = "|x=2,y=2|*CPU"} : () -> (tensor<i32>)
  func.return
}

// -----

// Check SPMD BroadcastGradientArgs op.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>, %arg1: tensor<8x3x3x3xf32>{tf._layout = "sharding_specs:x,unsharded,unsharded,unsharded, mesh:|x=2,y=2|*CPU"}) {
  // CHECK:      "tf_device.cluster"
  // CHECK:        %[[OUT1:.*]], %[[OUT2:.*]] = "tf.BroadcastGradientArgs"
  // CHECK-SAME:   _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3", "sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
  // CHECK-SAME:   (tensor<4xi32>, tensor<4xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  // CHECK-NEXT:   tf_device.return
  // CHECK-SAME:   %[[OUT1]]
  %0 = "tf_device.cluster"() ({
    %dimension = "tf.Const"() { value = dense<-1> : tensor<i64>, _layout = ["sharding_specs:scalar, mesh:|x=2,y=2|*CPU"]} : () -> tensor<i64>
    %v2 = "tf.Sum"(%arg1, %dimension) {keep_dims = true}: (tensor<8x3x3x3xf32>, tensor<i64>) -> tensor<8x3x3x1xf32>
    %s1 = "tf.Shape"(%arg1): (tensor<8x3x3x3xf32>) -> tensor<4xi32>
    %s2 = "tf.Shape"(%v2): (tensor<8x3x3x1xf32>) -> tensor<4xi32>
    %b1, %b2 = "tf.BroadcastGradientArgs"(%s1, %s2): (tensor<4xi32>, tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>)
    tf_device.return %b1 : tensor<4xi32>
  }) {_mesh = "|x=2,y=2|*CPU"} : () -> (tensor<i32>)
  func.return
}



