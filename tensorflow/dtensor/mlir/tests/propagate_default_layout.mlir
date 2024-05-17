// RUN: dtensor-opt %s -split-input-file -dtensor-propagate-default-layout | FileCheck %s

// Check that layouts attributes in function arguments are converted to layout
// ops.

// CHECK-LABEL: module @test_convert_arguments
module @test_convert_arguments {
// CHECK: func.func @main
// CHECK-SAME:  %arg[[ARG_0:[a-z0-9]*]]: tensor<i32>
// CHECK-SAME:  %arg[[ARG_1:[a-z0-9]*]]: tensor<i32>
func.func @main(
  %arg1: tensor<i32>{ tf._layout = "sharding_specs:scalar mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"},
  %arg2: tensor<i32>{ tf._layout = "sharding_specs:scalar mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"}) -> (tensor<i32>) {
  // CHECK:      %[[ARG1_OUT:[a-z0-9]*]] = "tf.DTensorLayout"(%arg[[ARG_1]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs: mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>
  // CHECK-SAME: dtensor.from_arg_index = [[ARG_1]]
  // CHECK-SAME: (tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: %[[ARG0_OUT:[a-z0-9]*]] = "tf.DTensorLayout"(%arg[[ARG_0]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs: mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>
  // CHECK-SAME: dtensor.from_arg_index = [[ARG_0]]
  // CHECK-SAME: (tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: "tf.A"(%[[ARG0_OUT]], %[[ARG1_OUT]])
  // CHECK-NEXT: "tf.B"(%[[ARG1_OUT]])
  // CHECK-NEXT: "tf.C"(%[[ARG0_OUT]])
  %1 = "tf.A"(%arg1, %arg2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  "tf.B"(%arg2) : (tensor<i32>) -> ()
  "tf.C"(%arg1) : (tensor<i32>) -> ()
  func.return %1 : tensor<i32>
}
}

// -----

// Check that layouts attributes in function outputs are converted to layout
// ops.

// CHECK-LABEL: module @test_convert_outputs
module @test_convert_outputs {
// CHECK-NEXT: func @main
func.func @main() -> (tensor<i32>{tf._default_layout = "sharding_specs:scalar mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"}) {
  // CHECK:      %[[A_OUT:.*]] = "tf.A"() : () -> tensor<i32>
  // CHECK-NEXT: %[[LAYOUT_A_OUT:.*]] = "tf.DTensorLayout"(%[[A_OUT]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs: mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>
  // CHECK-SAME: (tensor<i32>) -> tensor<i32>
  %1 = "tf.A"() : () -> tensor<i32>
  func.return %1 : tensor<i32>
}
}

// -----

// Check that layouts attributes of operations are correclty converted to layout
// op.

// CHECK-LABEL: module @test_convert_operations
module @test_convert_operations {
// CHECK-NEXT: func @main
func.func @main() -> (tensor<i32>) {
  // CHECK:      %[[A_OUT:.*]] = "tf.A"()
  // CHECK-NEXT: "tf.DTensorLayout"(%[[A_OUT]])
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs: mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>
  %1 = "tf.A"() {_layout = ["sharding_specs:scalar mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]} : () -> tensor<i32>
  func.return %1 : tensor<i32>
}
}

// -----

// Check that when creating DTensorLayout for operations, dtensor.from_arg_index is preserved.

// CHECK-LABEL: module @test_preserve_from_arg_index
module @test_preserve_from_arg_index {
// CHECK-NEXT: func @main
func.func @main() -> (tensor<i32>) {
  // CHECK:      %[[A_OUT:.*]] = "tf.A"()
  // CHECK-NEXT: "tf.DTensorLayout"(%[[A_OUT]])
  // CHECK-SAME: dtensor.from_arg_index = 3
  %1 = "tf.A"() {dtensor.from_arg_index = 3, _layout = ["sharding_specs:scalar mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]} : () -> tensor<i32>
  func.return %1 : tensor<i32>
}
}

// -----

// CHECK-LABEL: module @test_convert_resource_args
module @test_convert_resource_args {
// Check that resource typed arg with layouts are correctly converted to DTesnorLayout with global shape.
// CHECK-NEXT: func @main
func.func @main(%arg0: tensor<!tf_type.resource<tensor<4x2xf32>>>{ tf._layout = "sharding_specs:x,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"}) -> tensor<i32> {
  // CHECK:      "tf.DTensorLayout"(%arg0)
  // CHECK-SAME: global_shape = #tf_type.shape<4x2>
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:x,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>
  %1 = "tf.A"() {_layout = ["sharding_specs:scalar mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]} : () -> tensor<i32>
  func.return %1 : tensor<i32>
}
}

// -----

// CHECK-LABEL: module @test_convert_variant_args
module @test_convert_variant_args {
// Check that variant typed arg with layouts are correctly converted to DTesnorLayout.
// CHECK-NEXT: func @main
func.func @main(%arg0: tensor<!tf_type.variant<tensor<4x4xi32>>>{ tf._layout = "sharding_specs:x,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"}) -> tensor<i32> {
  // CHECK:      "tf.DTensorLayout"(%arg0)
  // CHECK-SAME: layout = #dtensor.layout<sharding_specs:x,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>
  %1 = "tf.A"() {_layout = ["sharding_specs:scalar mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]} : () -> tensor<i32>
  func.return %1 : tensor<i32>
}
}
