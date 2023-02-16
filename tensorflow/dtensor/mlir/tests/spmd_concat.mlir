// RUN: dtensor-opt -- %s -split-input-file -dtensor-annotate-global-shape -dtensor-layout-propagation-v2 -dtensor-spmd-expansion -verify-diagnostics | FileCheck %s

// Check that a dim sharded on all Concat inputs (which is not the concat dim)
// produces output layout with the same dim sharded.
// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]: tensor<2x16x32xf32>
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]: tensor<6x16x32xf32>
func.func @main(%arg0: tensor<2x32x32xf32> {tf._layout = "sharding_specs:unsharded,x,unsharded, mesh:|x=2,y=2|*TPU"},
           %arg1: tensor<6x32x32xf32> {tf._layout = "sharding_specs:unsharded,x,unsharded, mesh:|x=2,y=2|*TPU"}) {
  // CHECK:       "tf_device.cluster"
  // CHECK-NEXT:    %[[AXIS:.*]] = "tf.Const"()
  // CHECK-NEXT:    %[[CONCAT_OUT:.*]] = "tf.ConcatV2"(%[[ARG0]], %[[ARG1]], %[[AXIS]])
  // CHECK-SAME:      _layout = ["sharding_specs:unsharded,x,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
  // CHECK-SAME:      (tensor<2x16x32xf32>, tensor<6x16x32xf32>, tensor<i32>) -> tensor<8x16x32xf32>
  // CHECK:         tf_device.return
  // CHECK-SAME:      tensor<8x16x32xf32>
  %0 = "tf_device.cluster"() ({
    %cst = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<2x32x32>, layout = #dtensor.layout<sharding_specs:unsharded,x,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<2x32x32xf32>) -> tensor<2x32x32xf32>
    %2 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<6x32x32>, layout = #dtensor.layout<sharding_specs:unsharded,x,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<6x32x32xf32>) -> tensor<6x32x32xf32>
    %3 = "tf.ConcatV2"(%1, %2, %cst) : (tensor<2x32x32xf32>, tensor<6x32x32xf32>, tensor<i32>) -> tensor<8x32x32xf32>
    %4 = "tf.Identity"(%3) : (tensor<8x32x32xf32>) -> tensor<8x32x32xf32>
    tf_device.return %4 : tensor<8x32x32xf32>
  }) {_mesh="|x=2,y=2|*TPU"} : () -> (tensor<8x32x32xf32>)
  func.return
}

// -----

// Check that if the concat dim is sharded in any Concat inputs, then that dim
// is replicated in the output layout.
// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]: tensor<8x4x32xf32>
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]: tensor<8x2x32xf32>
// CHECK-SAME: %[[ARG2:[a-z0-9]*]]: tensor<8x4x32xf32>
func.func @main(%arg0: tensor<8x4x32xf32> {tf._layout = "sharding_specs:unsharded,unsharded,unsharded, mesh:|x=4|*TPU"},
           %arg1: tensor<8x8x32xf32> {tf._layout = "sharding_specs:unsharded,x,unsharded, mesh:|x=4|*TPU"},
           %arg2: tensor<8x16x32xf32> {tf._layout = "sharding_specs:unsharded,x,unsharded, mesh:|x=4|*TPU"}) {
  // CHECK:       "tf_device.cluster"
  // CHECK-NEXT:    %[[AXIS:.*]] = "tf.Const"()
  // CHECK-NEXT:    %[[ARG1_RELAYOUT:.*]] = "tf.DTensorAllGather"(%[[ARG1]])
  // CHECK-SAME:      _layout = ["sharding_specs:unsharded,unsharded,unsharded, mesh:|x=4|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
  // CHECK-SAME:      input_layout = #dtensor.layout<sharding_specs:unsharded,x,unsharded, mesh:|x=4|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK-SAME:      output_layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded, mesh:|x=4|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK-SAME:      (tensor<8x2x32xf32>) -> tensor<8x8x32xf32>
  // CHECK-NEXT:    %[[ARG2_RELAYOUT:.*]] = "tf.DTensorAllGather"(%[[ARG2]])
  // CHECK-SAME:      _layout = ["sharding_specs:unsharded,unsharded,unsharded, mesh:|x=4|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
  // CHECK-SAME:      input_layout = #dtensor.layout<sharding_specs:unsharded,x,unsharded, mesh:|x=4|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK-SAME:      output_layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded, mesh:|x=4|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK-SAME:      (tensor<8x4x32xf32>) -> tensor<8x16x32xf32>
  // CHECK-NEXT:    %[[CONCAT_OUT:.*]] = "tf.ConcatV2"(%[[ARG0]], %[[ARG1_RELAYOUT]], %[[ARG2_RELAYOUT]], %[[AXIS]])
  // CHECK-SAME:      _layout = ["sharding_specs:unsharded,unsharded,unsharded, mesh:|x=4|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
  // CHECK-SAME:      (tensor<8x4x32xf32>, tensor<8x8x32xf32>, tensor<8x16x32xf32>, tensor<i32>) -> tensor<8x28x32xf32>
  // CHECK:         tf_device.return
  // CHECK-SAME:      tensor<8x28x32xf32>
  %0 = "tf_device.cluster"() ({
    %cst = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<8x4x32>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded, mesh:|x=4|*TPU>} : (tensor<8x4x32xf32>) -> tensor<8x4x32xf32>
    %2 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x8x32>, layout = #dtensor.layout<sharding_specs:unsharded,x,unsharded, mesh:|x=4|*TPU>} : (tensor<8x8x32xf32>) -> tensor<8x8x32xf32>
    %3 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<8x16x32>, layout = #dtensor.layout<sharding_specs:unsharded,x,unsharded, mesh:|x=4|*TPU>} : (tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
    %4 = "tf.ConcatV2"(%1, %2, %3, %cst) : (tensor<8x4x32xf32>, tensor<8x8x32xf32>, tensor<8x16x32xf32>, tensor<i32>) -> tensor<8x28x32xf32>
    %5 = "tf.Identity"(%4) : (tensor<8x28x32xf32>) -> tensor<8x28x32xf32>
    tf_device.return %5 : tensor<8x28x32xf32>
  }) {_mesh="|x=4|*TPU"} : () -> (tensor<8x28x32xf32>)
  func.return
}

// -----

// Check that dims sharded on any Concat inputs (which is not the concat dim,
// and does not conflict with any other sharding) produces output layout with
// the same dims sharded.
// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]: tensor<4x4x32xf32>
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]: tensor<8x8x16xf32>
// CHECK-SAME: %[[ARG2:[a-z0-9]*]]: tensor<4x16x16xf32>
func.func @main(%arg0: tensor<8x4x32xf32> {tf._layout = "sharding_specs:x,unsharded,unsharded, mesh:|x=2,y=2|*TPU"},
           %arg1: tensor<8x8x32xf32> {tf._layout = "sharding_specs:unsharded,unsharded,y, mesh:|x=2,y=2|*TPU"},
           %arg2: tensor<8x16x32xf32> {tf._layout = "sharding_specs:x,unsharded,y, mesh:|x=2,y=2|*TPU"}) {
  // CHECK:       "tf_device.cluster"
  // CHECK-NEXT:    %[[AXIS:.*]] = "tf.Const"()
  // CHECK-NEXT:    %[[ARG0_RELAYOUT:.*]] = "tf.DTensorAllScatter"(%[[ARG0]])
  // CHECK-SAME:      _layout = ["sharding_specs:x,unsharded,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
  // CHECK-SAME:      input_layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK-SAME:      output_layout = #dtensor.layout<sharding_specs:x,unsharded,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK-SAME:      (tensor<4x4x32xf32>) -> tensor<4x4x16xf32>
  // CHECK-NEXT:    %[[ARG1_RELAYOUT:.*]] = "tf.DTensorAllScatter"(%[[ARG1]])
  // CHECK-SAME:      _layout = ["sharding_specs:x,unsharded,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
  // CHECK-SAME:      input_layout = #dtensor.layout<sharding_specs:unsharded,unsharded,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK-SAME:      output_layout = #dtensor.layout<sharding_specs:x,unsharded,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK-SAME:      (tensor<8x8x16xf32>) -> tensor<4x8x16xf32>
  // CHECK-NEXT:    %[[CONCAT_OUT:.*]] = "tf.ConcatV2"(%[[ARG0_RELAYOUT]], %[[ARG1_RELAYOUT]], %[[ARG2]], %[[AXIS]])
  // CHECK-SAME:      _layout = ["sharding_specs:x,unsharded,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
  // CHECK-SAME:      (tensor<4x4x16xf32>, tensor<4x8x16xf32>, tensor<4x16x16xf32>, tensor<i32>) -> tensor<4x28x16xf32>
  // CHECK:         tf_device.return
  // CHECK-SAME:      tensor<4x28x16xf32>
  %0 = "tf_device.cluster"() ({
    %cst = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<8x4x32>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<8x4x32xf32>) -> tensor<8x4x32xf32>
    %2 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x8x32>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,y, mesh:|x=2,y=2|*TPU>} : (tensor<8x8x32xf32>) -> tensor<8x8x32xf32>
    %3 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<8x16x32>, layout = #dtensor.layout<sharding_specs:x,unsharded,y, mesh:|x=2,y=2|*TPU>} : (tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
    %4 = "tf.ConcatV2"(%1, %2, %3, %cst) : (tensor<8x4x32xf32>, tensor<8x8x32xf32>, tensor<8x16x32xf32>, tensor<i32>) -> tensor<8x28x32xf32>
    %5 = "tf.Identity"(%4) : (tensor<8x28x32xf32>) -> tensor<8x28x32xf32>
    tf_device.return %5 : tensor<8x28x32xf32>
  }) {_mesh="|x=2,y=2|*TPU"} : () -> (tensor<8x28x32xf32>)
  func.return
}

// -----

// Check that any dims with conflicting sharding across the Concat inputs are
// deduplicated and the output layout is replicated in those dims.
// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]: tensor<8x4x32xf32>
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]: tensor<16x8x16xf32>
func.func @main(%arg0: tensor<8x8x32xf32> {tf._layout = "sharding_specs:unsharded,x,unsharded, mesh:|x=2,y=2|*TPU"},
           %arg1: tensor<16x8x32xf32> {tf._layout = "sharding_specs:unsharded,unsharded,x, mesh:|x=2,y=2|*TPU"}) {
  // CHECK:       "tf_device.cluster"
  // CHECK-NEXT:    %[[AXIS:.*]] = "tf.Const"()
  // CHECK-NEXT:    %[[ARG0_RELAYOUT:.*]] = "tf.DTensorAllGather"(%[[ARG0]])
  // CHECK-SAME:      _layout = ["sharding_specs:unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
  // CHECK-SAME:      input_layout = #dtensor.layout<sharding_specs:unsharded,x,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK-SAME:      output_layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK-SAME:      (tensor<8x4x32xf32>) -> tensor<8x8x32xf32>
  // CHECK-NEXT:    %[[ARG1_RELAYOUT:.*]] = "tf.DTensorAllGather"(%[[ARG1]])
  // CHECK-SAME:      _layout = ["sharding_specs:unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
  // CHECK-SAME:      input_layout = #dtensor.layout<sharding_specs:unsharded,unsharded,x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK-SAME:      output_layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK-SAME:      (tensor<16x8x16xf32>) -> tensor<16x8x32xf32>
  // CHECK-NEXT:    %[[CONCAT_OUT:.*]] = "tf.ConcatV2"(%[[ARG0_RELAYOUT]], %[[ARG1_RELAYOUT]], %[[AXIS]])
  // CHECK-SAME:      _layout = ["sharding_specs:unsharded,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
  // CHECK-SAME:      (tensor<8x8x32xf32>, tensor<16x8x32xf32>, tensor<i32>) -> tensor<24x8x32xf32>
  // CHECK:         tf_device.return
  // CHECK-SAME:      tensor<24x8x32xf32>
  %0 = "tf_device.cluster"() ({
    %cst = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<8x8x32>, layout = #dtensor.layout<sharding_specs:unsharded,x,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<8x8x32xf32>) -> tensor<8x8x32xf32>
    %2 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<16x8x32>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,x, mesh:|x=2,y=2|*TPU>} : (tensor<16x8x32xf32>) -> tensor<16x8x32xf32>
    %3 = "tf.ConcatV2"(%1, %2, %cst) : (tensor<8x8x32xf32>, tensor<16x8x32xf32>, tensor<i32>) -> tensor<24x8x32xf32>
    %4 = "tf.Identity"(%3) : (tensor<24x8x32xf32>) -> tensor<24x8x32xf32>
    tf_device.return %4 : tensor<24x8x32xf32>
  }) {_mesh="|x=2,y=2|*TPU"} : () -> (tensor<24x8x32xf32>)
  func.return
}

// -----

// Check that if any Concat input is sharded on the concat dim, along with other
// inputs sharded on other dims, then relayout is correctly applied to those
// inputs and the order of DTensorAllScatter -> DTensorAllGather is correct.
// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]: tensor<4x4x32xf32>
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]: tensor<8x4x32xf32>
// CHECK-SAME: %[[ARG2:[a-z0-9]*]]: tensor<4x8x32xf32>
func.func @main(%arg0: tensor<8x4x32xf32> {tf._layout = "sharding_specs:x,unsharded,unsharded, mesh:|x=2,y=2|*TPU"},
           %arg1: tensor<8x8x32xf32> {tf._layout = "sharding_specs:unsharded,y,unsharded, mesh:|x=2,y=2|*TPU"},
           %arg2: tensor<8x16x32xf32> {tf._layout = "sharding_specs:x,y,unsharded, mesh:|x=2,y=2|*TPU"}) {
  // CHECK:       "tf_device.cluster"
  // CHECK-NEXT:    %[[AXIS:.*]] = "tf.Const"()
  // CHECK-NEXT:    %[[ARG1_SCATTER:.*]] = "tf.DTensorAllScatter"(%[[ARG1]])
  // CHECK-SAME:      _layout = ["sharding_specs:x,y,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
  // CHECK-SAME:      input_layout = #dtensor.layout<sharding_specs:unsharded,y,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK-SAME:      output_layout = #dtensor.layout<sharding_specs:x,y,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK-SAME:      (tensor<8x4x32xf32>) -> tensor<4x4x32xf32>
  // CHECK-NEXT:    %[[ARG1_RELAYOUT:.*]] = "tf.DTensorAllGather"(%[[ARG1_SCATTER]])
  // CHECK-SAME:      _layout = ["sharding_specs:x,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
  // CHECK-SAME:      input_layout = #dtensor.layout<sharding_specs:x,y,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK-SAME:      output_layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK-SAME:      (tensor<4x4x32xf32>) -> tensor<4x8x32xf32>
  // CHECK-NEXT:    %[[ARG2_RELAYOUT:.*]] = "tf.DTensorAllGather"(%[[ARG2]])
  // CHECK-SAME:      _layout = ["sharding_specs:x,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
  // CHECK-SAME:      input_layout = #dtensor.layout<sharding_specs:x,y,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK-SAME:      output_layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>
  // CHECK-SAME:      (tensor<4x8x32xf32>) -> tensor<4x16x32xf32>
  // CHECK-NEXT:    %[[CONCAT_OUT:.*]] = "tf.ConcatV2"(%[[ARG0]], %[[ARG1_RELAYOUT]], %[[ARG2_RELAYOUT]], %[[AXIS]])
  // CHECK-SAME:      _layout = ["sharding_specs:x,unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
  // CHECK-SAME:      (tensor<4x4x32xf32>, tensor<4x8x32xf32>, tensor<4x16x32xf32>, tensor<i32>) -> tensor<4x28x32xf32>
  // CHECK:         tf_device.return
  // CHECK-SAME:      tensor<4x28x32xf32>
  %0 = "tf_device.cluster"() ({
    %cst = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<8x4x32>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<8x4x32xf32>) -> tensor<8x4x32xf32>
    %2 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x8x32>, layout = #dtensor.layout<sharding_specs:unsharded,y,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<8x8x32xf32>) -> tensor<8x8x32xf32>
    %3 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<8x16x32>, layout = #dtensor.layout<sharding_specs:x,y,unsharded, mesh:|x=2,y=2|*TPU>} : (tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
    %4 = "tf.ConcatV2"(%1, %2, %3, %cst) : (tensor<8x4x32xf32>, tensor<8x8x32xf32>, tensor<8x16x32xf32>, tensor<i32>) -> tensor<8x28x32xf32>
    %5 = "tf.Identity"(%4) : (tensor<8x28x32xf32>) -> tensor<8x28x32xf32>
    tf_device.return %5 : tensor<8x28x32xf32>
  }) {_mesh="|x=2,y=2|*TPU"} : () -> (tensor<8x28x32xf32>)
  func.return
}
