// RUN: dtensor-opt -- %s -split-input-file -dtensor-annotate-global-shape -dtensor-spmd-expansion -verify-diagnostics | FileCheck %s

module @test_spmd {
func.func @main() {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.A"() : () -> tensor<i32>
    %2 = "tf.B"() : () -> tensor<i32>
    // expected-error @+1 {{No attached layout found for op : tf.Add}}
    %3 = "tf.Add"(%1, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    tf_device.return %3 : tensor<i32>
  }) {_mesh = "TPU|x=2,y=1|*TPU"} : () -> (tensor<i32>)
  func.return
}
}
// -----
module @test_spmd_malformed_layouts {
// Check that ops with malformed layouts are disallowed.
func.func @main() {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.A"() : () -> tensor<i32>
    %2 = "tf.B"() : () -> tensor<i32>
    // expected-error @+1 {{Expected 2 items but found}}
    %3 = "tf.Add"(%1, %2) {_layout = [",,"]}: (tensor<i32>, tensor<i32>) -> tensor<i32>
    tf_device.return %3 : tensor<i32>
  }) {_mesh = "TPU|x=2,y=1|*TPU"} : () -> (tensor<i32>)
  func.return
}
}
// -----
module @test_spmd_operands_without_layouts {
// Check operands without layouts are disallowed.
func.func @main(%arg0: tensor<i32>) {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.A"() : () -> tensor<2x2xi32>
    %2 = "tf.B"() : () -> tensor<2x2xi32>
    // expected-error @+1 {{input layout of elementwise op must be known before SPMD expansion}}
    %3 = "tf.Add"(%1, %2) {_layout = ["sharding_specs:x,y, TPU|x=2,y=1|*TPU"]}: (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    tf_device.return %3 : tensor<2x2xi32>
  }) {_mesh = "TPU|x=2,y=1|*TPU"} : () -> (tensor<2x2xi32>)
  func.return
}
}
// -----

// Check SPMD is skipped for layouts with XLA SPMD mesh.
//
// Arguments and ops and Retvals should remain in global shape.
module @test_spmd_skipped_for_xla {
func.func @main(%arg0: tensor<8x8xi32> {tf._layout = "sharding_specs:x,unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7|use_xla_spmd"}) -> (tensor<8x8xi32>) {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<8x8>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7|use_xla_spmd>} : (tensor<8x8xi32>) -> tensor<8x8xi32>
    %2 = "tf.Identity"(%1) {_global_shape = [#tf_type.shape<8x8>], device = ""} : (tensor<8x8xi32>) -> tensor<8x8xi32>
    %3= "tf.DTensorLayout"(%2) {global_shape = #tf_type.shape<8x8>, layout = #dtensor.layout<sharding_specs:unsharded,x, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7|use_xla_spmd>} : (tensor<8x8xi32>) -> tensor<8x8xi32>
    tf_device.return %3 : tensor<8x8xi32>
  }) {_mesh = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7|use_xla_spmd"} : () -> tensor<8x8xi32>
  return %0 : tensor<8x8xi32>
}
}
// -----

// Check that elementwise batch parallel op SPMD expansion.
// CHECK-LABEL: module @test_spmd_batch_parallel
module @test_spmd_batch_parallel {
func.func @main(
  %arg0: tensor<2x2xi32> { tf._layout = "sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|*TPU"},
  %arg1: tensor<2x2xi32> { tf._layout = "sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|*TPU"}) {
  %0 = "tf_device.cluster"() ({
    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:   "tf.Add"
    // CHECK-SAME:   _layout = ["sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
    %3 = "tf.Add"(%arg0, %arg1) {_layout = ["sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|*TPU"]}: (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    tf_device.return %3 : tensor<2x2xi32>
  }) {_mesh = "TPU|x=2,y=2|*TPU"} : () -> (tensor<2x2xi32>)
  func.return
}
}
// -----

// Check tf.Add SPMD with sharded inputs/outputs
// CHECK-LABEL: module @test_spmd_sharded_inputs
module @test_spmd_sharded_inputs {
func.func @main(
  %arg0: tensor<2x2xi32> { tf._layout = "sharding_specs:x,y, mesh:TPU|x=2,y=2|*TPU"},
  %arg1: tensor<2x2xi32> { tf._layout = "sharding_specs:x,y, mesh:TPU|x=2,y=2|*TPU"}) {
  // CHECK:        "tf_device.cluster"
  // CHECK-NEXT:      %[[ADD_OUT:.*]] = "tf.Add"
  // CHECK-NEXT:      tf_device.return
  // CHECK-SAME:      %[[ADD_OUT]]
  %0 = "tf_device.cluster"() ({
    %3 = "tf.Add"(%arg0, %arg1) {_layout = ["sharding_specs:x,y, mesh:TPU|x=2,y=2|*TPU"]}: (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    tf_device.return %3 : tensor<2x2xi32>
  }) {_mesh = "TPU|x=2,y=2|*TPU"} : () -> (tensor<i32>)
  func.return
}
}

// -----

// Check tf.Neg Op SPMD.
// CHECK-LABEL: module @test_spmd_neg_op
module @test_spmd_neg_op {
func.func @main(
  %arg0: tensor<2x2xi32> { tf._layout = "sharding_specs:x,y, mesh:TPU|x=2,y=2|*TPU"}) {
  // CHECK:        "tf_device.cluster"
  // CHECK-NEXT:      %[[NEG_OUT:.*]] = "tf.Neg"
  // CHECK-NEXT:      tf_device.return
  // CHECK-SAME:      %[[NEG_OUT]]
  %0 = "tf_device.cluster"() ({
    %2 = "tf.Neg"(%arg0) {_layout = ["sharding_specs:x,y, mesh:TPU|x=2,y=2|*TPU"]}: (tensor<2x2xi32>) -> tensor<2x2xi32>
    tf_device.return %2 : tensor<2x2xi32>
  }) {_mesh = "TPU|x=2,y=2|*TPU"} : () -> (tensor<i32>)
  func.return
}
}

// -----

// Check replicated tf.Const op SPMD.
// CHECK-LABEL: module @test_spmd_const_op
module @test_spmd_const_op {
func.func @main(%arg0: tensor<i32>) {
  // CHECK:        "tf_device.cluster"
  // CHECK-NEXT:      %[[A_OUT:.*]] = "tf.Const"
  // CHECK-NEXT:      %[[B_OUT:.*]] = "tf.Const"
  // CHECK-NEXT:      %[[ADD_OUT:.*]] = "tf.Add"(%[[A_OUT]], %[[B_OUT]])
  // CHECK-NEXT:      tf_device.return
  // CHECK-SAME:      %[[ADD_OUT]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const"() {_layout = ["sharding_specs:unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"], value=dense<1> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
    %2 = "tf.Const"() {_layout = ["sharding_specs:unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"], value=dense<1> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
    %3 = "tf.Add"(%1, %2) {_layout = ["sharding_specs:unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]}: (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
    tf_device.return %3 : tensor<1x1xi32>
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> tensor<1x1xi32>
  func.return
}
}
// -----

  // Check sharded tf.Const op SPMD.
// CHECK-LABEL: module @test_spmd_const_op_sharded
module @test_spmd_const_op_sharded {
  // CHECK: func @main
  // CHECK-SAME: %[[ARG0:[a-z0-9]*]]: tensor<i32>
  func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<1>}) {
    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:      %[[A_OUT:.*]] = "tf.Const"
    // CHECK-NEXT:      %[[A_SLICE:[0-9]*]] = "tf.DTensorAllScatter"(%[[A_OUT]])
    // CHECK-SAME:      input_layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
    // CHECK-SAME:      output_layout = #dtensor.layout<sharding_specs:x, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
    // CHECK-NEXT:      %[[IDENTITY_OUT:[0-9]*]] = "tf.IdentityN"(%[[A_SLICE]])
    // CHECK-NEXT:      %[[B_OUT:.*]] = "tf.Const"
    // CHECK-NEXT:      %[[ADD_OUT:[0-9]*]] = "tf.Add"(%[[IDENTITY_OUT]], %[[B_OUT]])
    // CHECK-NEXT:      tf_device.return
    // CHECK-SAME:      %[[ADD_OUT]]
    %0 = "tf_device.cluster"() ({
      %1 = "tf.Const"() {value = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
      %2 = "tf.DTensorLayout"(%1) {global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:x, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2xi32>) -> (tensor<2xi32>)
      %3 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<1>, layout = #dtensor.layout<sharding_specs:scalar, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<i32>) -> (tensor<i32>)
      %5 = "tf.Add"(%2, %4) {_layout = ["sharding_specs:x, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]}: (tensor<2xi32>, tensor<i32>) -> tensor<2xi32>
      tf_device.return %5 : tensor<2xi32>
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> tensor<2xi32>
  func.return
  }
}

// -----

// Check sharded tf.Const op SPMD with splat.
// CHECK-LABEL: module @test_spmd_const_op_sharded_with_splat
module @test_spmd_const_op_sharded_with_splat {
func.func @main(%arg0: tensor<i32>) {
  // CHECK:        "tf_device.cluster"
  // CHECK-NEXT:      %[[CONST_OUT:.*]] = "tf.Const"() {[[BEFORE_ATTR:.*]]value = dense<1> : tensor<1xi32>[[AFTER_ATTR:.*]]} : () -> tensor<1xi32>
  // CHECK-NEXT:      tf_device.return
  %0 = "tf_device.cluster"() ({
   %1 = "tf.Const"() {_layout = ["sharding_specs:x, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"], value=dense<1>: tensor<2xi32>} : () -> tensor<2xi32>
   tf_device.return %1 : tensor<2xi32>
 }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> tensor<2xi32>
 func.return
}
}
// -----

// Check replicated tf.BroadcastTo op SPMD.
// CHECK-LABEL: module @test_spmd_broadcast_replicated
module @test_spmd_broadcast_replicated {
func.func @main(%arg0: tensor<3xi32>) {
  // CHECK:       "tf.BroadcastTo"
  // CHECK-SAME:  tensor<3xi32>, tensor<2xi64>) -> tensor<3x3xi32>
  %0 = "tf_device.cluster"() ({
    %1 = arith.constant dense<[3, 3]> : tensor<2xi32>
    %2 = "tf.DTensorLayout"(%1) {global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2xi32>) -> (tensor<2xi32>)
    %3 = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<3>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<3xi32>) -> (tensor<3xi32>)
    %4 = "tf.BroadcastTo"(%3, %2) {_layout = ["sharding_specs:unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"], device = ""} : (tensor<3xi32>, tensor<2xi32>) -> tensor<3x3xi32>
    tf_device.return %4 : tensor<3x3xi32>
 }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> tensor<3x3xi32>
 func.return
}
}
// -----

// Check replicated tf.range op SPMD.
// CHECK-LABEL: module @test_spmd_range
module @test_spmd_range {
func.func @main() {
  // CHECK:       "tf.Range"
  // CHECK-SAME:  tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3xi32>
  %0 = "tf_device.cluster"() ({
    %0 = arith.constant dense<0> : tensor<i32>
    %1 = arith.constant dense<3> : tensor<i32>
    %2 = arith.constant dense<1> : tensor<i32>
    %3 = "tf.Range"(%0, %1, %2) {_layout = ["sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"], device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3xi32>
    tf_device.return %3 : tensor<3xi32>
 }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> tensor<3xi32>
 func.return
}
}

// -----

// Check tf.AssignVariable op SPMD
// CHECK-LABEL: module @test_spmd_assign_var
module @test_spmd_assign_var {
func.func @main(%arg0: tensor<32x32xi32> { tf._layout = "sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}, %arg1: tensor<!tf_type.resource> { tf._layout = "sharding_specs: mesh:||||"}) {
  // CHECK:        "tf_device.cluster"
  // CHECK-NEXT:     "tf.AssignVariableOp"
  // CHECK-NEXT:     tf_device.return
  // CHECK-NEXT:     _inferred_resource_indices = dense<1> : vector<1xi32>
  // CHECK-SAME:     _inferred_resource_layouts
  // CHECK-SAME:     "sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"
  "tf_device.cluster"() ({
    "tf.AssignVariableOp"(%arg1, %arg0) {dtype = i32} : (tensor<!tf_type.resource>, tensor<32x32xi32>) -> ()
    tf_device.return
 }) { _mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> ()
 func.return
}
}
// -----

// Check tf.Softmax op SPMD where last dimension is not sharded.
// CHECK-LABEL: module @test_spmd_softmax_last_dim_unsharded
module @test_spmd_softmax_last_dim_unsharded {
func.func @main(%arg0: tensor<32x32xf32> { tf._layout = "sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}) {
  // CHECK:        "tf_device.cluster"
  // CHECK-NEXT:     "tf.Softmax"
  // CHECK-NEXT:     tf_device.return
  "tf_device.cluster"() ({
    "tf.Softmax"(%arg0) {_layout = ["sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]} : (tensor<32x32xf32>) -> (tensor<32x32xf32>)
    tf_device.return
 }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> ()
 func.return
}
}

// -----

// Check tf.Softmax op with rank 3.
// CHECK-LABEL: module @test_spmd_softmax_rank_3
module @test_spmd_softmax_rank_3 {
func.func @main(%arg0: tensor<32x32x32xf32> { tf._layout = "sharding_specs:x,y,unsharded, mesh:TPU|x=2,y=2,z=2|*TPU"}) {
  // CHECK:        "tf_device.cluster"
  // CHECK-NEXT:     "tf.Softmax"
  // CHECK-NEXT:     tf_device.return
  "tf_device.cluster"() ({
    "tf.Softmax"(%arg0) {_layout = ["sharding_specs:x,y,unsharded, mesh:TPU|x=2,y=2,z=2|*TPU"]} : (tensor<32x32x32xf32>) -> (tensor<32x32x32xf32>)
    tf_device.return
 }) {_mesh = "TPU|x=2,y=2,z=2|*TPU"} : () -> ()
 func.return
}
}

// -----

// Check SPMD expansion of softmax op with non-sharded last dimension.
// CHECK-LABEL: module @test_spmd_softmax_last_dim_unsharded1
module @test_spmd_softmax_last_dim_unsharded1 {
func.func @main(%arg0: tensor<32x32xf32> { tf._layout = "sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}) {
  // CHECK:        "tf_device.cluster"
  // CHECK-NEXT:     "tf.LogSoftmax"
  // CHECK-NEXT:     tf_device.return
  "tf_device.cluster"() ({
    "tf.LogSoftmax"(%arg0) {_layout = ["sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]} : (tensor<32x32xf32>) -> (tensor<32x32xf32>)
    tf_device.return
 }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> ()
 func.return
}
}
// -----

// Check SPMD expansion of SoftMax op.
// CHECK-LABEL: module @test_spmd_softmax
module @test_spmd_softmax {
func.func @main(%arg0: tensor<32x32x32xf32> { tf._layout = "sharding_specs:x,y,unsharded, mesh:TPU|x=2,y=2,z=2|*TPU"}) {
  // CHECK:        "tf_device.cluster"
  // CHECK-NEXT:     "tf.LogSoftmax"
  // CHECK-NEXT:     tf_device.return
  "tf_device.cluster"() ({
    "tf.LogSoftmax"(%arg0) {_layout = ["sharding_specs:x,y,unsharded, mesh:TPU|x=2,y=2,z=2|*TPU"]} : (tensor<32x32x32xf32>) -> (tensor<32x32x32xf32>)
    tf_device.return
 }) {_mesh = "TPU|x=2,y=2,z=2|*TPU"} : () -> ()
 func.return
}
}
// -----

// Check that Softmax op with last dimension sharded is supported on TPU's.
// CHECK-LABEL: module @test_spmd_softmax_last_dim_sharded
module @test_spmd_softmax_last_dim_sharded {
func.func @main(%arg0: tensor<32x32x32xf32> { tf._layout = "sharding_specs:x,y,z, mesh:TPU|x=2,y=2,z=2|*TPU"}) {
  "tf_device.cluster"() ({
    "tf.Softmax"(%arg0) {_layout = ["sharding_specs:x,y,z, mesh:TPU|x=2,y=2,z=2|*TPU"]} : (tensor<32x32x32xf32>) -> (tensor<32x32x32xf32>)
    tf_device.return
 }) {_mesh = "TPU|x=2,y=2,z=2|*TPU"} : () -> ()
 func.return
}
}

// -----

// Check that random uniform op with incompatible shape is disallowed.
module @test_spmd_random_op_with_incomplete_shape_disallowed {
func.func @main(%arg0: tensor<i32>) {
  %0 = "tf_device.cluster"() ({
    // %1 = "tf.Const"() {_layout = ["sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"], device = "", value = dense<16> : tensor<2xi32>} : () -> tensor<2xi32>
    // %2 = "tf.Const"() {_layout = ["sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"], device = "", value = dense<[123, 321]> : tensor<2xi32>} : () -> tensor<2xi32>
    %1 = arith.constant dense<[16]> : tensor<1xi32>
    %2 = arith.constant dense<[2, 1]> : tensor<2xi32>
    // expected-error @+1 {{Sharding dimension of random op does not match rank of the random op}}
    %3 = "tf.StatelessRandomUniform"(%1, %2) {_layout = ["sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"], device = ""} : (tensor<1xi32>, tensor<2xi32>) -> tensor<16x16xf32>
    tf_device.return %3 : tensor<16x16xf32>
 }) { _mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> tensor<16x16xf32>
 func.return
}
}

// -----

// Check Resource Apply op SPMD.
// CHECK-LABEL: module @test_spmd_resource
module @test_spmd_resource {
// CHECK: func @main
// CHECK-SAME: %arg0: tensor<f32>
// CHECK-SAME: %arg1: tensor<1x1xf32>
// CHECK-SAME: tf._layout = "sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"
// CHECK-SAME: %arg2: tensor<!tf_type.resource>
// CHECK-SAME: tf._layout = "sharding_specs: mesh:||||"
func.func @main(
  %arg0: tensor<f32>,
  %arg1: tensor<2x2xf32>{ tf._layout = "sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"},
  %arg2: tensor<!tf_type.resource> { tf._layout = "sharding_specs: mesh:||||"}) {
  // CHECK:        "tf_device.cluster"
  // CHECK-NEXT:     "tf.ResourceApplyGradientDescent"(%arg2, %arg0, %arg1)
  // CHECK-NEXT:     tf_device.return
  "tf_device.cluster"() ({
    "tf.ResourceApplyGradientDescent"(%arg2, %arg0, %arg1) {_layout = ["sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"], use_locking = false} : (tensor<!tf_type.resource>, tensor<f32>, tensor<2x2xf32>) -> ()
    tf_device.return
 }) { _mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> ()
 func.return
}
}

// -----

// Check that function inputs are modified to reflect local input shapes.
// CHECK-LABEL: module @test_spmd_inputs_have_local_shapes
module @test_spmd_inputs_have_local_shapes {
// CHECK: func @main
// CHECK-SAME: %arg0: tensor<f32>
// CHECK-SAME: %arg1: tensor<1x1xf32>
// CHECK-SAME: tf._layout = "sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"
// CHECK-SAME: %arg2: tensor<!tf_type.resource>
// CHECK-SAME: tf._layout = "sharding_specs: mesh:||||"
func.func @main(
  %arg0: tensor<f32>,
  %arg1: tensor<2x2xf32>{ tf._layout = "sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"},
  %arg2: tensor<!tf_type.resource> { tf._layout = "sharding_specs: mesh:||||"}) {
  "tf_device.cluster"() ({
    tf_device.return
 }) {_mesh = "mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> ()
 func.return
}
}

// -----

// Check that tf_device.Cluster op return values are updated to reflect local
// shape.
// CHECK-LABEL: module @test_spmd_returned_values_have_local_shapes
module @test_spmd_returned_values_have_local_shapes {
// CHECK: func @main
// CHECK-SAME: %arg0: tensor<f32>
// CHECK-SAME: %arg1: tensor<1x1xf32>
// CHECK-SAME: tf._layout = "sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"
// CHECK-SAME: %arg2: tensor<!tf_type.resource>
// CHECK-SAME: tf._layout = "sharding_specs: mesh:||||"
func.func @main(
  %arg0: tensor<f32>,
  %arg1: tensor<2x2xf32>{ tf._layout = "sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"},
  %arg2: tensor<!tf_type.resource> { tf._layout = "sharding_specs: mesh:||||"}) {
  "tf_device.cluster"() ({
    // CHECK:      tf_device.return
    // CHECK-SAME: tensor<1x1xf32>
    tf_device.return %arg1 : tensor<2x2xf32>
 }) {_mesh = "mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> (tensor<2x2xf32>)
 func.return
}
}
// -----

// Check that function signature as well as return types of callsite operations
// are updated to reflect local shape.
// CHECK-LABEL: module @test_spmd_return_types_have_local_shapes_at_callsite
module @test_spmd_return_types_have_local_shapes_at_callsite {
// CHECK: func @main
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]: tensor<f32>
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]: tensor<1x1xf32>
// CHECK-SAME: tf._layout = "sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"
// CHECK-SAME: %[[ARG2:[a-z0-9]*]]: tensor<!tf_type.resource>
// CHECK-SAME: tf._layout = "sharding_specs: mesh:||||"
func.func @main(
  %arg0: tensor<f32>,
  %arg1: tensor<2x2xf32>{ tf._layout = "sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"},
  %arg2: tensor<!tf_type.resource> { tf._layout = "sharding_specs: mesh:||||"}) {
  "tf_device.cluster"() ({
    // CHECK:     "tf.StatefulPartitionedCall"(%[[ARG1]])
    // CHECK-SAME: (tensor<1x1xf32>) -> tensor<1x1xf32>
    %0 = "tf.StatefulPartitionedCall"(%arg1) {config = "", config_proto = "", executor_type = "", f = @pcall_func} : (tensor<2x2xf32>) -> (tensor<2x2xf32>)
    tf_device.return %arg1 : tensor<2x2xf32>
 }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> (tensor<2x2xf32>)
 func.return
}

// CHECK: func @pcall_func
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]: tensor<1x1xf32>
func.func @pcall_func(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: return %[[ARG0]] : tensor<1x1xf32>
  func.return %arg0 : tensor<2x2xf32>
}
}
// -----

// Check DTensorLayout ops are removed after SPMD Expansion.
// CHECK-LABEL: module @test_spmd_dtensor_layout_ops_are_removed
module @test_spmd_dtensor_layout_ops_are_removed {
// CHECK: func @main
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
    %2 = "tf.DTensorLayout"(%1) {global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2x2xi32>) -> (tensor<2x2xi32>)
    %3 = "tf.Const"() {value = dense<[[1, 1], [2, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2x2xi32>) -> (tensor<2x2xi32>)
    %5 = "tf.Add"(%2, %4): (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    %6 = "tf.DTensorLayout"(%5) {global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2x2xi32>) -> (tensor<2x2xi32>)
    tf_device.return %6 : tensor<2x2xi32>
  }) {_mesh = "mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> (tensor<2x2xi32>)
  func.return
}
}
// -----
// CHECK-LABEL: module @test_spmd_neg_op1
module @test_spmd_neg_op1 {
// CHECK: func @main
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]: tensor<i32>
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]: tensor<2x2xf32>
func.func @main(
  %arg0: tensor<i32>, %arg1: tensor<2x2xf32>) {
  "tf_device.cluster"() ({
    // CHECK:       %[[ARG1_SLICE:[0-9]*]] = "tf.DTensorAllScatter"(%[[ARG1]])
    // CHECK-SAME:  input_layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
    // CHECK-SAME:  output_layout = #dtensor.layout<sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
    // CHECK-NEXT:  %[[NEG_OUT:[0-9]*]] = "tf.Neg"(%[[ARG1_SLICE]])
    // CHECK-NEXT:  tf_device.return
    // CHECK-SAME:  %[[NEG_OUT]]
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2x2xf32>) -> (tensor<2x2xf32>)
    %2 = "tf.Neg"(%1) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %3 = "tf.DTensorLayout"(%2) {global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2x2xf32>) -> (tensor<2x2xf32>)
    tf_device.return %3: tensor<2x2xf32>
 }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> (tensor<2x2xf32>)
 func.return
}
}
// -----

// A super tricky case where the DTensorLayout is out of the tf_device.cluster and somewhat gets casted to BlockArgument with a wild argument number 3.
// CHECK-LABEL: module @test_spmd_var_input
module @test_spmd_var_input {
// CHECK: func @main
// CHECK:       %arg1: tensor<!tf_type.resource<tensor<1xf32>>>
func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>},
  %arg1: tensor<!tf_type.resource<tensor<2xf32>>> {tf._global_shape = #tf_type.shape<2>, tf._layout = "empty_layout", tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"},
  %arg2: tensor<2xf32> {tf._global_shape = #tf_type.shape<2>,
    tf._layout = "sharding_specs:x, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1",
    tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"}) {
  %0 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<!tf_type.resource<tensor<2xf32>>>) -> tensor<!tf_type.resource<tensor<2xf32>>>
  "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg2) {_global_shape = [#tf_type.shape<2>], global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2xf32>) -> tensor<2xf32>
    "tf.AssignVariableOp"(%0, %1) {_global_shape = [], device = ""} : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
    tf_device.return
  }) {_mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"} : () -> ()
  func.return
}
}

// -----
// A super tricky case where the DTensorLayout is out of the tf_device.cluster and somewhat gets casted to BlockArgument with a wild argument number 3.
// CHECK-LABEL: module @test_spmd_assigned_value_is_input
module @test_spmd_assigned_value_is_input {
  // CHECK: func @main
  // CHECK:          "tf.VarHandleOp"()
  // CHECK-SAME:      _layout = ["sharding_specs:x, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"]
  func.func @main(%arg0: tensor<2xf32> {tf._global_shape = #tf_type.shape<2>,
      tf._layout = "sharding_specs:x, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1",
      tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"}) {
    // %1 =  "tf.DTensorLayout"(%0) {global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<!tf_type.resource<tensor<2xf32>>>) -> tensor<!tf_type.resource<tensor<2xf32>>>
    "tf_device.cluster"() ({
      %0 = "tf.VarHandleOp"() {_global_shape = [#tf_type.shape<2>], allowed_devices = [], container = "", device = "", shared_name = ""} : () -> tensor<!tf_type.resource<tensor<2xf32>>> 
      %1 = "tf.DTensorLayout"(%0) {_global_shape = [#tf_type.shape<2>], global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:x, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<!tf_type.resource<tensor<2xf32>>>) -> tensor<!tf_type.resource<tensor<2xf32>>>
      %2 = "tf.DTensorLayout"(%arg0) {_global_shape = [#tf_type.shape<2>], global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:x, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2xf32>) -> tensor<2xf32>
      "tf.AssignVariableOp"(%1, %2) {_global_shape = [], device = ""} : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
      tf_device.return
    }) {_mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"} : () -> ()
    func.return
  }
}

// -----

// Check to ensure that the local shape of resource-type arguments are not double-calculated if they are assigned to a tensor value wihtin the function.
// CHECK-LABEL: module @test_spmd_var_arg_local_shapes
module @test_spmd_var_arg_local_shapes {
// CHECK: func @main
// CHECK-SAME: %arg0: tensor<i32>
// CHECK-SAME: %arg1: tensor<1x4xf32>
// CHECK-SAME: %arg2: tensor<!tf_type.resource<tensor<1x4xf32>>>
func.func @main(
  %arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>},
  %arg1: tensor<2x4xf32> {
    tf._global_shape = #tf_type.shape<2x4>,
    tf._layout = "sharding_specs:x,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1",
    tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"},
  %arg2: tensor<*x!tf_type.resource<tensor<2x4xf32>>> {
    tf._global_shape = #tf_type.shape<2x4>,
    tf._layout = "sharding_specs:x,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1",
    tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"})
  -> (tensor<2x4xf32> {tf._global_shape = #tf_type.shape<2x4>}) {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg2) {_global_shape = [#tf_type.shape<*>], global_shape = #tf_type.shape<2x4>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<*x!tf_type.resource<tensor<2x4xf32>>>) -> tensor<*x!tf_type.resource<tensor<2x4xf32>>>
    %2 = "tf.DTensorLayout"(%arg1) {_global_shape = [#tf_type.shape<2x4>], global_shape = #tf_type.shape<2x4>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    "tf.AssignVariableOp"(%1, %2) {_global_shape = [], device = ""} : (tensor<*x!tf_type.resource<tensor<2x4xf32>>>, tensor<2x4xf32>) -> ()
    %3 = "tf.ReadVariableOp"(%1) {_global_shape = [#tf_type.shape<2x4>], device = ""} : (tensor<*x!tf_type.resource<tensor<2x4xf32>>>) -> tensor<2x4xf32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<2x4>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %5 = "tf.Identity"(%4) {_global_shape = [#tf_type.shape<2x4>], device = ""} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %6 = "tf.DTensorLayout"(%5) {global_shape = #tf_type.shape<2x4>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    tf_device.return %6 : tensor<2x4xf32>
  }) {_mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"} : () -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}
}
// -----

// Check SPMD expansion of Cumsum op with sharding on axis dimension, should
// produce a replicated layout on that axis dimension, with allgather and
// allscatter for intermediate layout computation.
// CHECK-LABEL: module @test_spmd_cumsum_op
module @test_spmd_cumsum_op {
// CHECK: func @main
func.func @main(%arg0: tensor<32x32xf32> { tf._layout = "sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}) {
  // CHECK:        "tf_device.cluster"
  // CHECK-NEXT:   "tf.DTensorAllGather"
  // CHECK-NEXT:   "tf.Const"
  // CHECK-NEXT:   "tf.Cumsum"
  // CHECK-NEXT:   "tf.DTensorAllScatter"
  // CHECK-NEXT:    tf_device.return
  "tf_device.cluster"() ({
     %axis = "tf.Const"() { value = dense<0> : tensor<i64>, _layout = ["sharding_specs:scalar, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]} : () -> tensor<i64>
    "tf.Cumsum"(%arg0, %axis) {_layout = ["sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]} : (tensor<32x32xf32>, tensor<i64>) -> (tensor<32x32xf32>)
    tf_device.return
 }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> ()
 func.return
}
}

// -----

// Check SPMD expansion of Cumsum op with no sharding on axis dim. This should
// not produce an allscatter or allgather for intermediate layout computation
// since no relayouts are happening.
// CHECK-LABEL: module @test_spmd_cumsum_op_no_sharding_on_axis_dim
module @test_spmd_cumsum_op_no_sharding_on_axis_dim {
// CHECK: func @main
func.func @main(%arg0: tensor<32x32xf32> { tf._layout = "sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}) {
  // CHECK:        "tf_device.cluster"
  // CHECK-NEXT:   "tf.Const"
  // CHECK-NEXT:   "tf.Cumsum"
  // CHECK-NEXT:    tf_device.return
  "tf_device.cluster"() ({
     %axis = "tf.Const"() { value = dense<-1> : tensor<i64>, _layout = ["sharding_specs:scalar, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]} : () -> tensor<i64>
    "tf.Cumsum"(%arg0, %axis) {_layout = ["sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]} : (tensor<32x32xf32>, tensor<i64>) -> (tensor<32x32xf32>)
    tf_device.return
 }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> ()
 func.return
}
}
// -----

// Check Relayout for SparseTensors emits the appropriate ops required for relaying out a SparseTensor.
// We do this by doing a matmul (between a sparsetensor and a densetensor)
// tf.matmul (*,x) multiplied by (x,*) causes a relayout on the left operand.
// CHECK-LABEL: module @test_spmd_relayout_on_sparse_tensors
module @test_spmd_relayout_on_sparse_tensors {
// CHECK: func @main
func.func @main(%arg0: tensor<i32>, %arg1: tensor<4x16xf32> {tf._layout = "sharding_specs:unsharded,x, mesh:|batch=2,x=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"}, %arg2: tensor<?x2xi64>, %arg3: tensor<2xi64>, %arg4: tensor<?xf32>) -> tensor<8x16xf32> {
  // CHECK: %[[CST:.*]] = "tf.Const"
  // CHECK-NEXT: %[[DENSE_0:.*]] = "tf.SparseToDense"(%arg2, %arg3, %arg4, %[[CST]])
  // CHECK-NEXT: %[[RIGHT_OPERAND:.*]] = "tf.DTensorAllScatter"(%arg1)
  // CHECK-NEXT: %[[ONE:.*]] = "tf.DTensorAllGather"(%[[DENSE_0]])
  // CHECK-NEXT: %[[TWO:.*]] = "tf.ZerosLike"(%[[ONE]])
  // CHECK-NEXT: %[[THREE:.*]] = "tf.NotEqual"(%[[ONE]], %[[TWO]])
  // CHECK-NEXT: %[[WHERE:.*]] = "tf.Where"(%[[THREE]])
  // CHECK-NEXT: %[[GATHER:.*]] = "tf.GatherNd"(%[[ONE]], %[[WHERE]])
  // CHECK-NEXT: %[[SHAPE:.*]] = "tf.Shape"(%[[ONE]])
  // CHECK-NEXT: %[[CST_0:.*]] = "tf.Const"
  // CHECK-NEXT: %[[LEFT_OPERAND:.*]] = "tf.SparseToDense"(%[[WHERE]], %[[SHAPE]], %[[GATHER]], %[[CST_0]])
  // CHECK-NEXT: "tf.MatMul"(%[[LEFT_OPERAND]], %[[RIGHT_OPERAND]])
  %0 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<?x2>, layout = #dtensor.layout<sharding_specs:x,batch, mesh:|batch=2,x=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3>} : (tensor<?x2xi64>) -> tensor<?x2xi64>
  %1 = "tf_device.cluster"() ({
    %cst = "tf.Const"() {_global_shape = [#tf_type.shape<>], value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %2 = "tf.DTensorLayout"(%cst) {global_shape = #tf_type.shape<>, layout = #dtensor.layout<sharding_specs: mesh:|batch=2,x=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3>} : (tensor<f32>) -> tensor<f32>
    %3 = "tf.SparseToDense"(%0, %arg3, %arg4, %2) {_global_shape = [#tf_type.shape<8x4>]} : (tensor<?x2xi64>, tensor<2xi64>, tensor<?xf32>, tensor<f32>) -> tensor<8x4xf32>
    %4 = "tf.DTensorLayout"(%arg1) {_global_shape = [#tf_type.shape<4x16>], global_shape = #tf_type.shape<4x16>, layout = #dtensor.layout<sharding_specs:unsharded,x, mesh:|batch=2,x=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3>} : (tensor<4x16xf32>) -> tensor<4x16xf32>
    %5 = "tf.DTensorLayout"(%3) {_global_shape = [#tf_type.shape<8x4>], global_shape = #tf_type.shape<8x4>, layout = #dtensor.layout<sharding_specs:x,batch, mesh:|batch=2,x=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
    %6 = "tf.MatMul"(%5, %4) {_global_shape = [#tf_type.shape<8x16>], device = "", transpose_a = false, transpose_b = false} : (tensor<8x4xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
    %7 = "tf.DTensorLayout"(%6) {global_shape = #tf_type.shape<8x16>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|batch=2,x=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3>} : (tensor<8x16xf32>) -> tensor<8x16xf32>
    %8 = "tf.Identity"(%7) {_global_shape = [#tf_type.shape<8x16>], device = ""} : (tensor<8x16xf32>) -> tensor<8x16xf32>
    %9 = "tf.DTensorLayout"(%8) {global_shape = #tf_type.shape<8x16>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|batch=2,x=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3>} : (tensor<8x16xf32>) -> tensor<8x16xf32>
    tf_device.return %9 : tensor<8x16xf32>
  }) {_mesh = "|batch=2,x=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"} : () -> tensor<8x16xf32>
  func.return %1 : tensor<8x16xf32>
}
}

// -----

// Check that relayout uses all-to-all for unsharded,x to x,unsharded.
// CHECK-LABEL: module @test_relayout_using_all_to_all
module @test_relayout_using_all_to_all {
// CHECK: func @main
func.func @main(%arg0: tensor<i32>, %arg1: tensor<32x32xf32> { tf._layout = "sharding_specs:unsharded,x, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}) -> tensor<32x32xf32>  {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT: %[[CST:.*]] = "tf.Const"
  // CHECK-NEXT: %[[BIAS_ADD_OUT:.*]] = "tf.BiasAdd"(%arg1, %cst)
  // CHECK-NEXT: %[[ALL_TO_ALL_OUT:.*]] = "tf.DTensorAllToAll"(%[[BIAS_ADD_OUT]])
  // CHECK-SAME: input_layout = #dtensor.layout<sharding_specs:unsharded,x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1,/job:localhost/replica:0/task:0/device:TPU:2,/job:localhost/replica:0/task:0/device:TPU:3>
  // CHECK-SAME: output_layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1,/job:localhost/replica:0/task:0/device:TPU:2,/job:localhost/replica:0/task:0/device:TPU:3>
  // CHECK-NEXT: tf_device.return
  %0 = "tf_device.cluster"() ({
    %cst = "tf.Const"() {value = dense<0.000000e+00> : tensor<32xf32>, _layout = ["sharding_specs:x, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]} : () -> tensor<32xf32>
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<32x32>, layout = #dtensor.layout<sharding_specs:unsharded,x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1,/job:localhost/replica:0/task:0/device:TPU:2,/job:localhost/replica:0/task:0/device:TPU:3>} : (tensor<32x32xf32>) -> tensor<32x32xf32>
    %2 = "tf.BiasAdd"(%1, %cst) {global_shape = #tf_type.shape<32x32>} : (tensor<32x32xf32>, tensor<32xf32>) -> (tensor<32x32xf32>)
    %3 = "tf.DTensorLayout"(%2) {global_shape = #tf_type.shape<32x32>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1,/job:localhost/replica:0/task:0/device:TPU:2,/job:localhost/replica:0/task:0/device:TPU:3>} : (tensor<32x32xf32>) -> tensor<32x32xf32>
    tf_device.return %3 : tensor<32x32xf32>
 }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> (tensor<32x32xf32>)
 func.return %0 : tensor<32x32xf32>
}
}

// -----

// Check SPMD expansion of TensorListReserve replicated and TensorListSet with a sharded tensor emits a gather to replicated.
// CHECK-LABEL: module @test_spmd_tensor_list_reserve_replicated
module @test_spmd_tensor_list_reserve_replicated {
// CHECK: func @main
func.func @main(%arg0: tensor<i32>, %arg1: tensor<4x4xi32> {tf._layout = "sharding_specs:x,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"}) -> (tensor<4x4xi32>) {
  // CHECK:        "tf_device.cluster"
  // CHECK-NEXT:   "tf.Const"
  // CHECK-NEXT:   "tf.Const"
  // CHECK-NEXT:   "tf.Const"
  // CHECK-NEXT:   "tf.Const"
  // CHECK-NEXT:   "tf.Const"
  // CHECK-NEXT:   "tf.DTensorAllGather"
  // CHECK-NEXT:   "tf.Const"
  // CHECK-NEXT:   "tf.TensorListReserve"
  // CHECK-NEXT:   "tf.TensorListSetItem"
  // CHECK-NEXT:   "tf.TensorListGetItem"
  %0 = "tf_device.cluster"() ({
    %cst = "tf.Const"() {_global_shape = [#tf_type.shape<2>], value = dense<4> : tensor<2xi32>} : () -> tensor<2xi32>
    %1 = "tf.DTensorLayout"(%cst) {global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2xi32>) -> tensor<2xi32>
    %cst_0 = "tf.Const"() {_global_shape = [#tf_type.shape<2>], value = dense<4> : tensor<2xi32>} : () -> tensor<2xi32>
    %2 = "tf.DTensorLayout"(%cst_0) {global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2xi32>) -> tensor<2xi32>
    %cst_1 = "tf.Const"() {_global_shape = [#tf_type.shape<>], value = dense<4> : tensor<i32>} : () -> tensor<i32>
    %3 = "tf.DTensorLayout"(%cst_1) {global_shape = #tf_type.shape<>, layout = #dtensor.layout<sharding_specs: mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<i32>) -> tensor<i32>
    %cst_2 = "tf.Const"() {_global_shape = [#tf_type.shape<>], value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %4 = "tf.DTensorLayout"(%cst_2) {global_shape = #tf_type.shape<>, layout = #dtensor.layout<sharding_specs: mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<i32>) -> tensor<i32>
    %cst_3 = "tf.Const"() {_global_shape = [#tf_type.shape<>], value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %5 = "tf.DTensorLayout"(%cst_3) {global_shape = #tf_type.shape<>, layout = #dtensor.layout<sharding_specs: mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<i32>) -> tensor<i32>
    %6 = "tf.DTensorLayout"(%arg1) {_global_shape = [#tf_type.shape<4x4>], global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %7 = "tf.TensorListReserve"(%1, %3) {_global_shape = [#tf_type.shape<>], device = ""} : (tensor<2xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<4x4xi32>>>
    %8 = "tf.DTensorLayout"(%7) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<!tf_type.variant<tensor<4x4xi32>>>) -> tensor<!tf_type.variant<tensor<4x4xi32>>>
    %9 = "tf.TensorListSetItem"(%8, %4, %6) {_global_shape = [#tf_type.shape<>], device = ""} : (tensor<!tf_type.variant<tensor<4x4xi32>>>, tensor<i32>, tensor<4x4xi32>) -> tensor<!tf_type.variant<tensor<4x4xi32>>>
    %10 = "tf.DTensorLayout"(%9) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<!tf_type.variant<tensor<4x4xi32>>>) -> tensor<!tf_type.variant<tensor<4x4xi32>>>
    %11 = "tf.TensorListGetItem"(%10, %5, %2) {_global_shape = [#tf_type.shape<4x4>], device = ""} : (tensor<!tf_type.variant<tensor<4x4xi32>>>, tensor<i32>, tensor<2xi32>) -> tensor<4x4xi32>
    %12 = "tf.DTensorLayout"(%11) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %13 = "tf.Identity"(%12) {_global_shape = [#tf_type.shape<4x4>], device = ""} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    %14 = "tf.DTensorLayout"(%13) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<4x4xi32>) -> tensor<4x4xi32>
    tf_device.return %14 : tensor<4x4xi32>
  }) {_mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"} : () -> tensor<4x4xi32>
  func.return %0 : tensor<4x4xi32>
}
}

// -----

// Check SPMD expansion of DisableCopyOnRead has correct shape.
// CHECK-LABEL: module @test_spmd_disable_copy_on_read
module @test_spmd_disable_copy_on_read {
// CHECK: func @main
func.func @main(
  %arg0: tensor<i32>,
  %arg1: tensor<!tf_type.resource<tensor<4x8xi32>>> {tf._layout = "sharding_specs:x,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"}) -> () {
  // CHECK: "tf_device.cluster"
  // CHECK:   "tf.DisableCopyOnRead"(%arg1) {_global_shape = [], _layout = [], device = ""} : (tensor<!tf_type.resource<tensor<2x8xi32>>>) -> ()
  "tf_device.cluster"() ({
    %0 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<4x8>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<!tf_type.resource<tensor<4x8xi32>>>) -> tensor<!tf_type.resource<tensor<4x8xi32>>>
    "tf.DisableCopyOnRead"(%0) {device = ""} : (tensor<!tf_type.resource<tensor<4x8xi32>>>) -> ()
    tf_device.return
  }) {_mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"} : () -> ()
  func.return
}
}
// -----

// Check SPMD expansion of ScatterNd op output is the sharding of updates
// tensor.
// CHECK-LABEL: module @test_spmd_scatter_nd_op
module @test_spmd_scatter_nd_op {
// CHECK: func @main
func.func @main(%arg0: tensor<2x4x4xi32> {tf._layout = "sharding_specs:unsharded,x,unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7"}) -> (tensor<16x4x4xi32>) {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:   "tf.Const"()
  // CHECK-NEXT:   %[[INDICES:.*]] = "tf.Const"()
  // CHECK-NEXT:   %[[NEW_SHAPE:.*]] = "tf.Const"() {value = dense<[16, 2, 4]> : tensor<3xi32>} : () -> tensor<3xi32>
  // CHECK-NEXT:   "tf.ScatterNd"(%[[INDICES]], %arg0, %[[NEW_SHAPE]])
  %0 = "tf_device.cluster"() ({
    %shape = "tf.Const"() {_global_shape = [#tf_type.shape<3>], value = dense<[16, 4, 4]> : tensor<3xi32>} : () -> tensor<3xi32>
    %indices = "tf.Const"() {_global_shape = [#tf_type.shape<2x1>], value = dense<[[0], [15]]> : tensor<2x1xi32>} : () -> tensor<2x1xi32>
    %updates_with_layout = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<2x4x4>, layout = #dtensor.layout<sharding_specs:unsharded,x,unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7>} : (tensor<2x4x4xi32>) -> tensor<2x4x4xi32>
    %indices_with_layout= "tf.DTensorLayout"(%indices) {global_shape = #tf_type.shape<2x1>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7>} : (tensor<2x1xi32>) -> tensor<2x1xi32>
    %shape_with_layout = "tf.DTensorLayout"(%shape) {global_shape = #tf_type.shape<3>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7>} : (tensor<3xi32>) -> tensor<3xi32>
    %4 = "tf.ScatterNd"(%indices_with_layout, %updates_with_layout, %shape_with_layout) {_global_shape = [#tf_type.shape<16x4x4>], device = ""} : (tensor<2x1xi32>, tensor<2x4x4xi32>, tensor<3xi32>) -> tensor<16x4x4xi32>
    %5 = "tf.DTensorLayout"(%4) {global_shape = #tf_type.shape<16x4x4>, layout = #dtensor.layout<sharding_specs:unsharded,x,unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7>} : (tensor<16x4x4xi32>) -> tensor<16x4x4xi32>
    tf_device.return %5 : tensor<16x4x4xi32>
  }) {_mesh = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7"} : () -> tensor<16x4x4xi32>
  return %0 : tensor<16x4x4xi32>
}
}
// -----

// Check SPMD expansion of ScatterNd op indices is relayout to replicated.
// CHECK-LABEL: module @test_spmd_scatter_nd_op_indices_layout
module @test_spmd_scatter_nd_op_indices_layout {
// CHECK: func @main
func.func @main(%arg0: tensor<2x4x4xi32> {tf._layout = "sharding_specs:unsharded,x,unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7"},
                %arg1: tensor<2x1xi32> {tf._global_shape = #tf_type.shape<2x1>, tf._layout = "sharding_specs:x,unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7"}) -> (tensor<16x4x4xi32>) {
  // CHECK:   "tf_device.cluster"
  // CHECK:     "tf.DTensorAllGather"(%arg1)
  // CHECK:     "tf.ScatterNd"
  %0 = "tf_device.cluster"() ({
    %shape = "tf.Const"() {_global_shape = [#tf_type.shape<3>], value = dense<[16, 4, 4]> : tensor<3xi32>} : () -> tensor<3xi32>
    %updates_with_layout = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<2x4x4>, layout = #dtensor.layout<sharding_specs:unsharded,x,unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7>} : (tensor<2x4x4xi32>) -> tensor<2x4x4xi32>
    %indices_with_layout= "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<2x1>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7>} : (tensor<2x1xi32>) -> tensor<2x1xi32>
    %shape_with_layout = "tf.DTensorLayout"(%shape) {global_shape = #tf_type.shape<3>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7>} : (tensor<3xi32>) -> tensor<3xi32>
    %4 = "tf.ScatterNd"(%indices_with_layout, %updates_with_layout, %shape_with_layout) {_global_shape = [#tf_type.shape<16x4x4>], device = ""} : (tensor<2x1xi32>, tensor<2x4x4xi32>, tensor<3xi32>) -> tensor<16x4x4xi32>
    %5 = "tf.DTensorLayout"(%4) {global_shape = #tf_type.shape<16x4x4>, layout = #dtensor.layout<sharding_specs:unsharded,x,unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7>} : (tensor<16x4x4xi32>) -> tensor<16x4x4xi32>
    tf_device.return %5 : tensor<16x4x4xi32>
  }) {_mesh = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7"} : () -> tensor<16x4x4xi32>
  return %0 : tensor<16x4x4xi32>
}
}
// -----

// Check stateful random operations raise error.
module @test_spmd_error_on_stateful_random_op {
func.func @main(
  %arg0: tensor<2xi32> { tf._layout = "sharding_specs:unsharded, mesh:CPU|x=2,y=2|*CPU"}) {
  %0 = "tf_device.cluster"() ({
    // expected-error @+1 {{Stateful random operations are not supported in DTensor.}}
    %1 = "tf.RandomUniform"(%arg0) {_layout = ["sharding_specs:x,y, mesh:CPU|x=2,y=2|*CPU"]}: (tensor<2xi32>) -> tensor<4x4xf32>
    %2 = "tf.DTensorLayout"(%1) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:x,y, mesh:CPU|x=2,y=2|*CPU>} : (tensor<4x4xf32>) -> (tensor<4x4xf32>)
    tf_device.return %2 : tensor<4x4xf32>
  }) {_mesh = "CPU|x=2,y=2|*CPU"} : () -> (tensor<4x4xf32>)
  func.return
}
}

// -----

// Check stateful random operations raise error.
module @test_spmd_error_on_stateful_random_op1 {
func.func @main(
  %arg0: tensor<2xi32> { tf._layout = "sharding_specs:unsharded, mesh:CPU|x=2,y=2|*CPU"},
  %arg1: tensor<1xi32> { tf._layout = "sharding_specs:unsharded, mesh:CPU|x=2,y=2|*CPU"},
  %arg2: tensor<1xi32> { tf._layout = "sharding_specs:unsharded, mesh:CPU|x=2,y=2|*CPU"}
) {
  %0 = "tf_device.cluster"() ({
    // expected-error @+1 {{Stateful random operations are not supported in DTensor.}}
    %1 = "tf.RandomUniformInt"(%arg0, %arg1, %arg2) {_layout = ["sharding_specs:x,y, mesh:CPU|x=2,y=2|*CPU"]}: (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4x4xi32>
    %2 = "tf.DTensorLayout"(%1) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:x,y, mesh:CPU|x=2,y=2|*TPU>} : (tensor<4x4xi32>) -> (tensor<4x4xi32>)
    tf_device.return %2 : tensor<4x4xi32>
  }) {_mesh = "CPU|x=2,y=2|*CPU"} : () -> (tensor<4x4xi32>)
  func.return
}
}