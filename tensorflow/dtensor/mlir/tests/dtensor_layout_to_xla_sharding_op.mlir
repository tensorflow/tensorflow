// RUN: dtensor-opt %s -dtensor-layout-to-xla-sharding-op | FileCheck %s

// CHECK-LABEL: @check_layouts_are_converted_to_xla_sharding_op
func.func @check_layouts_are_converted_to_xla_sharding_op(
  %arg0: tensor<8x8xi32> {tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0|use_xla_spmd"}) -> tensor<8x8xi32> {
  // CHECK:      [[tensor1:%[0-9]+]] = "tf.Identity"(%arg0)
  // CHECK:      [[tensor2:%[0-9]+]] = "tf.XlaSharding"([[tensor1]])
  // CHECK-SAME: _XlaSharding = ""
  // CHECK: return [[tensor2]]
  %1 = "tf.Identity"(%arg0) {_global_shape = [#tf_type.shape<8x8>], device = ""} : (tensor<8x8xi32>) -> tensor<8x8xi32>
  %2 = "tf.DTensorLayout"(%1) {global_shape = #tf_type.shape<8x8>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0|use_xla_spmd>} : (tensor<8x8xi32>) -> tensor<8x8xi32>
  return %2 : tensor<8x8xi32>
}

// -----

// CHECK-LABEL: @no_xla_sharding_op_for_block_arg
func.func @no_xla_sharding_op_for_block_arg(
  %arg0: tensor<8x8xi32> {tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0|use_xla_spmd"}) -> tensor<8x8xi32> {
  // CHECK-NOT: "tf.DTensorLayout"
  // CHECK-NOT: "tf.XlaSharding"
  // CHECK: return %arg0
  %1 = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<8x8>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0|use_xla_spmd>} : (tensor<8x8xi32>) -> tensor<8x8xi32>
  return %1 : tensor<8x8xi32>
}

// -----

// CHECK-LABEL: @no_xla_sharding_op_for_const_input
func.func @no_xla_sharding_op_for_const_input() -> tensor<8x8xi32> {
  // CHECK: [[tensor:%[a-z0-9]+]] = "tf.Const"
  // CHECK-NOT: "tf.DTensorLayout"
  // CHECK-NOT: "tf.XlaSharding"
  // CHECK: return [[tensor]]
  %cst = "tf.Const"() {value = dense<-3> : tensor<i32>} : () -> tensor<8x8xi32>
  %1 = "tf.DTensorLayout"(%cst) {global_shape = #tf_type.shape<8x8>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0|use_xla_spmd>} : (tensor<8x8xi32>) -> tensor<8x8xi32>
  return %1 : tensor<8x8xi32>
}

// -----

// CHECK-LABEL: @no_xla_sharding_op_for_const_foldable_input
func.func @no_xla_sharding_op_for_const_foldable_input() -> tensor<8x8xi32> {
  // CHECK: [[tensor:%[a-z0-9]+]] = "tf.Const"
  // CHECK-NOT: "tf.DTensorLayout"
  // CHECK-NOT: "tf.XlaSharding"
  // CHECK-NOT: "tf.Reshape"
  // CHECK: return [[tensor]]
  %cst = "tf.Const"() {value = dense<-3> : tensor<i32>} : () -> tensor<8x8xi32>
  %1 = "tf.DTensorLayout"(%cst) {global_shape = #tf_type.shape<8x8>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0|use_xla_spmd>} : (tensor<8x8xi32>) -> tensor<8x8xi32>
  %shape = arith.constant dense<8> : tensor<2xi32>
  %2 = "tf.Reshape"(%1, %shape) {_global_shape = [#tf_type.shape<8x8>], device = ""} : (tensor<8x8xi32>, tensor<2xi32>) -> tensor<8x8xi32>
  %3 = "tf.DTensorLayout"(%2) {global_shape = #tf_type.shape<8x8>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0|use_xla_spmd>} : (tensor<8x8xi32>) -> tensor<8x8xi32>
  return %3 : tensor<8x8xi32>
}
