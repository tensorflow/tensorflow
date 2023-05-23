// RUN: dtensor-opt %s -split-input-file -dtensor-annotate-global-shape -dtensor-layout-propagation -dtensor-spmd-expansion -verify-diagnostics | FileCheck %s --dump-input=fail

// Check Fill op on sharded default input as argument.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>, %arg1: tensor<2xi32> {tf._layout = "sharding_specs:unsharded, mesh:CPU|x=2,y=2|*CPU" }, %arg2: tensor<f32> ) -> (tensor<?x?xf32> {tf._default_layout = "sharding_specs:x,unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"}) {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:   "tf.Const"
  // CHECK-SAME:   dense<[2, 1]>
  // CHECK-NEXT:   "tf.Div"
  // CHECK-NEXT:   "tf.Fill"
  // CHECK-SAME:   _layout = ["sharding_specs:x,unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
  // CHECK-SAME:   tensor<2xi32>, tensor<f32>) -> tensor<?x?xf32>
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Fill"(%arg1, %arg2) : (tensor<2xi32>, tensor<f32>) -> tensor<?x?xf32>
    tf_device.return %1 : tensor<?x?xf32>
  }) {} : () -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

// Check Fill op on sharded default input as ConstOp.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>, %arg1: tensor<2xi32> {tf._layout = "sharding_specs:unsharded, mesh:CPU|x=2,y=2|*CPU"}, %arg2: tensor<f32> ) -> (tensor<8x1xf32>{
  tf._default_layout = "sharding_specs:x,unsharded, mesh:CPU|x=2,y=2|*CPU"}) {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:   "tf.Const"
  // CHECK-NEXT:   dense<[2, 1]>
  // CHECK-NEXT:   "tf.Div"
  // CHECK-NEXT:   "tf.Fill"
  // CHECK-SAME:   {_layout = ["sharding_specs:x,unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]}
  // CHECK-SAME:   tensor<2xi32>, tensor<f32>) -> tensor<?x?xf32>
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const"() {value = dense<[8, 1]> : tensor<2xi32>, _layout = ["sharding_specs:unsharded, mesh:CPU|x=2,y=2|*CPU"]} : () -> tensor<2xi32>
    %2 = "tf.Fill"(%1, %arg2) {device = ""} : (tensor<2xi32>, tensor<f32>) -> tensor<8x1xf32>
    tf_device.return %2 : tensor<8x1xf32>
  }) {} : () -> tensor<8x1xf32>
  func.return %0 : tensor<8x1xf32>
}

// -----

// Check tf.Fill op with incompatible layout disallowed.
func.func @main(%arg0: tensor<i32>, %arg1: tensor<2xi32> {tf._layout = "sharding_specs:unsharded, mesh:CPU|x=2,y=2|*CPU"}, %arg2: tensor<f32> ) -> (tensor<8x1xf32>{
  // expected-error @+4 {{The sharding spec for axis 0 splits among 3 values}}
  tf._default_layout = "sharding_specs:x,unsharded, mesh:CPU|x=3,y=2|*CPU"}) {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const"() {value = dense<[8, 1]> : tensor<2xi32>, _layout = ["sharding_specs:unsharded, mesh:CPU|x=3,y=2|*CPU"]} : () -> tensor<2xi32>
    %2 = "tf.Fill"(%1, %arg2) {device = ""} : (tensor<2xi32>, tensor<f32>) -> tensor<8x1xf32>
    tf_device.return %2 : tensor<8x1xf32>
  }) {} : () -> tensor<8x1xf32>
  func.return %0 : tensor<8x1xf32>
}
