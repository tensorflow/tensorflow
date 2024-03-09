// RUN: dtensor-opt %s -split-input-file -dtensor-annotate-global-shape -dtensor-spmd-expansion -verify-diagnostics | FileCheck %s --dump-input=fail

// Batch matmul, no batch dimensions
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1xi32>,
           %arg1: tensor<2x4xi32> {tf._layout = "sharding_specs:unsharded,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"},
           %arg2: tensor<4x2xi32> {tf._layout = "sharding_specs:y,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}) {
  // CHECK:    "tf_device.cluster"
  // CHECK:      %[[MATMUL_OUT:.*]] = "tf.BatchMatMulV2"(%arg1, %arg2)
  // CHECK-SAME: (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK:      %[[GROUP_ID:.*]] = "tf.Const"() <{value = dense<{{.*}}> : tensor<2x2xi32>}> : () -> tensor<2x2xi32>
  // CHECK:      %[[SUM_OUT:.*]] = "tf.DTensorAllReduce"(%[[MATMUL_OUT]], %[[GROUP_ID]])
  // CHECK-SAME: reduce_op = "Add"
  // CHECK-SAME: _layout = ["sharding_specs:unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
  // CHECK-SAME: (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK-NEXT: tf_device.return
  // CHECK-SAME: %[[SUM_OUT]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<2x4>, layout = #dtensor.layout<sharding_specs:unsharded,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2x4xi32>) -> tensor<2x4xi32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<4x2>, layout = #dtensor.layout<sharding_specs:y,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<4x2xi32>) -> tensor<4x2xi32>
    %3 = "tf.BatchMatMulV2"(%1, %2) {adj_x = false, adj_y = false}: (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<2x2xi32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2x2xi32>) -> tensor<2x2xi32>
    tf_device.return %4 : tensor<2x2xi32>
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> (tensor<i32>)
  func.return
}

// -----

// Batch matmul with batch dims
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1xi32>,
           %arg1: tensor<8x2x4xi32> {tf._layout = "sharding_specs:x,unsharded,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"},
           %arg2: tensor<8x4x2xi32> {tf._layout = "sharding_specs:x,y,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}) {
  // CHECK:    "tf_device.cluster"
  // CHECK:      %[[MATMUL_OUT:.*]] = "tf.BatchMatMulV2"
  // CHECK-SAME: (tensor<4x2x2xi32>, tensor<4x2x2xi32>) -> tensor<4x2x2xi32>
  // CHECK:      %[[GROUP_ID:.*]] = "tf.Const"() <{value = dense<{{.*}}> : tensor<2x2xi32>}> : () -> tensor<2x2xi32>
  // CHECK:      %[[SUM_OUT:.*]] = "tf.DTensorAllReduce"(%[[MATMUL_OUT]], %[[GROUP_ID]])
  // CHECK-SAME: reduce_op = "Add"
  // CHECK-SAME: _layout = ["sharding_specs:x,unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
  // CHECK-SAME: (tensor<4x2x2xi32>, tensor<2x2xi32>) -> tensor<4x2x2xi32>
  // CHECK-NEXT: tf_device.return
  // CHECK-SAME: %[[SUM_OUT]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x2x4>, layout = #dtensor.layout<sharding_specs:x,unsharded,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<8x2x4xi32>) -> tensor<8x2x4xi32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<8x4x2>, layout = #dtensor.layout<sharding_specs:x,y,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<8x4x2xi32>) -> tensor<8x4x2xi32>
    %3 = "tf.BatchMatMulV2"(%1, %2) {adj_x = false, adj_y = false}: (tensor<8x2x4xi32>, tensor<8x4x2xi32>) -> tensor<8x2x2xi32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<8x2x2>, layout = #dtensor.layout<sharding_specs:x,unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<8x2x2xi32>) -> tensor<8x2x2xi32>
    tf_device.return %4 : tensor<8x2x2xi32>
  }) { _mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> (tensor<i32>)
  func.return
}

// -----

// Batch matmul, with incompatible dimensions
func.func @main(%arg0: tensor<1xi32>,
           %arg1: tensor<2x4xi32> {tf._layout = "sharding_specs:x,y, mesh:TPU|x=2,y=2,z=2|*TPU"},
           %arg2: tensor<4x2xi32> {tf._layout = "sharding_specs:z,unsharded, mesh:TPU|x=2,y=2,z=2|*TPU"}) {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<2x4>, layout = #dtensor.layout<sharding_specs:x,y, mesh:TPU|x=2,y=2,z=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7>} : (tensor<2x4xi32>) -> tensor<2x4xi32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<4x2>, layout = #dtensor.layout<sharding_specs:z,unsharded, mesh:TPU|x=2,y=2,z=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7>} : (tensor<4x2xi32>) -> tensor<4x2xi32>
    // expected-error @+1 {{Contracting dimension for matmul has sharding dimension y for the left input and z for the right input which are not equal.}}
    %3 = "tf.BatchMatMulV2"(%1, %2) {adj_x = false, adj_y = false}: (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<2x2xi32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:TPU|x=2,y=2,z=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7>} : (tensor<2x2xi32>) -> tensor<2x2xi32>
    tf_device.return %4 : tensor<2x2xi32>
  }) {_layout = ["sharding_specs:x,y, mesh:TPU|x=2,y=2,z=2|*TPU"], _mesh = "TPU|x=2,y=2|*TPU"} : () -> (tensor<i32>)
  func.return
}

// -----

// Regular matmul
func.func @main(%arg0: tensor<1xi32>,
           %arg1: tensor<2x4xi32> {tf._layout = "sharding_specs:unsharded,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"},
           %arg2: tensor<4x2xi32> {tf._layout = "sharding_specs:y,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}) {
  // CHECK:      "tf_device.cluster"
  // CHECK:        %[[MATMUL_OUT:.*]] = "tf.MatMul"(%arg1, %arg2)
  // CHECK-SAME:   (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK-NEXT:   %[[GROUP_ID:.*]] = "tf.Const"()
  // CHECK-NEXT:   %[[SUM_OUT:.*]] = "tf.DTensorAllReduce"(%[[MATMUL_OUT]], %[[GROUP_ID]])
  // CHECK-SAME: reduce_op = "Add"
  // CHECK-SAME:   _layout = ["sharding_specs:unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
  // CHECK-SAME:   (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK-NEXT:   tf_device.return
  // CHECK-SAME:   %[[SUM_OUT]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<2x4>, layout = #dtensor.layout<sharding_specs:unsharded,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2x4xi32>) -> tensor<2x4xi32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<4x2>, layout = #dtensor.layout<sharding_specs:y,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<4x2xi32>) -> tensor<4x2xi32>
    %3 = "tf.MatMul"(%1, %2) {transpose_a = false, transpose_b = false}: (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<2x2xi32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<2x2xi32>) -> tensor<2x2xi32>
    tf_device.return %4 : tensor<2x2xi32>
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> (tensor<i32>)
  func.return
}

// -----

// Regular MatMul with one operand sharded
func.func @main(%arg0: tensor<i32>,
           %arg1: tensor<4x4xf32> {tf._layout = "sharding_specs:unsharded,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"},
           %arg2: tensor<4x3xf32> {tf._layout = "sharding_specs:unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}) -> tensor<4x3xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT: %[[B_SLICE:[0-9]*]] = "tf.DTensorAllScatter"(%arg2)
  // CHECK:      %[[MATMUL_OUT:.*]] = "tf.MatMul"(%arg1, %[[B_SLICE]])
  // CHECK:      %[[COLL_OUT:.*]] = "tf.DTensorAllReduce"
  // CHECK-NEXT: tf_device.return
  // CHECK-SAME: %[[COLL_OUT]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:unsharded,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<4x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<4x3xf32>) -> tensor<4x3xf32>
    %3 = "tf.MatMul"(%1, %2) {transpose_a = false, transpose_b = false} : (tensor<4x4xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<4x3>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<4x3xf32>) -> tensor<4x3xf32>
    tf_device.return %4 : tensor<4x3xf32>
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> tensor<4x3xf32>
  func.return %0 : tensor<4x3xf32>
}

// -----

// y,x . x,y -> *,y
// We unshard %arg1
func.func @main(%arg0: tensor<i32>,
           %arg1: tensor<4x4xf32> {tf._layout = "sharding_specs:y,x, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"},
           %arg2: tensor<4x4xf32> {tf._layout = "sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}) -> tensor<4x4xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[GATHERED:[0-9]*]] = "tf.DTensorAllGather"(%arg1)
  // CHECK-NEXT: %[[MATMUL_RESULT:[0-9]*]] = "tf.MatMul"(%[[GATHERED]], %arg2)
  // CHECK:      %[[FINAL_REDUCE:[0-9]*]] = "tf.DTensorAllReduce"(%[[MATMUL_RESULT]], %cst)
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:y,x, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = "tf.MatMul"(%1, %2) {transpose_a = false, transpose_b = false} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:unsharded,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    tf_device.return %4 : tensor<4x4xf32>
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

// *,x . x,* -> *,y
// We should slice arg2 before matmul rather than slicing the result.
func.func @main(%arg0: tensor<i32>,
           %arg1: tensor<4x4xf32> {tf._layout = "sharding_specs:unsharded,x, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"},
           %arg2: tensor<4x4xf32> {tf._layout = "sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}) -> tensor<4x4xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[SLICE:[0-9]*]] = "tf.DTensorAllScatter"(%arg2)
  // CHECK-NEXT: %[[MATMUL_RESULT:[0-9]*]] = "tf.MatMul"(%arg1, %[[SLICE]])
  // CHECK:      %[[FINAL_REDUCE:[0-9]*]] = "tf.DTensorAllReduce"(%[[MATMUL_RESULT]], %cst)
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:unsharded,x, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = "tf.MatMul"(%1, %2) {transpose_a = false, transpose_b = false} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:unsharded,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    tf_device.return %4 : tensor<4x4xf32>
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

// x,y . *,y -> x,y
// We unshard %arg1 on y.
func.func @main(%arg0: tensor<i32>,
           %arg1: tensor<4x4xf32> {tf._layout = "sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"},
           %arg2: tensor<4x4xf32> {tf._layout = "sharding_specs:unsharded,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}) -> tensor<4x4xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[GATHERED:[0-9]*]] = "tf.DTensorAllGather"(%arg1)
  // CHECK-NEXT: %[[MATMUL_RESULT:[0-9]*]] = "tf.MatMul"(%[[GATHERED]], %arg2)
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:unsharded,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = "tf.MatMul"(%1, %2) {transpose_a = false, transpose_b = false} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<4x4>, layout = #dtensor.layout<sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    tf_device.return %4 : tensor<4x4xf32>
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}
