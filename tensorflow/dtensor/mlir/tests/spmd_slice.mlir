// RUN: dtensor-opt %s -split-input-file -dtensor-annotate-global-shape -dtensor-layout-propagation-v2 -dtensor-spmd-expansion -verify-diagnostics | FileCheck %s --dump-input=always

// Check SPMD of splice op with replicated input.

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>, %arg1: tensor<2x4xf32> {tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU"}) -> tensor<2x2xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:        "tf.Slice"(%arg1, %cst, %cst_1)
  // CHECK-SAME:     _layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
  // CHECK-SAME:     (tensor<2x4xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xf32>
  // CHECK:        tf_device.return
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<2x4>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU>} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32>
    %3 = "tf.Const"() {value = dense<[-1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
    %4 = "tf.Slice"(%1, %2, %3) : (tensor<2x4xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xf32>
    tf_device.return %4 : tensor<2x2xf32>
  }) {_mesh = "|x=2,y=2|*CPU", _layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU"]} : () -> (tensor<2x2xf32>)
  func.return %0 : tensor<2x2xf32>
}

// -----

// Check that the slice on sharded x dimension is from 1, which requires a relayout to a fully replicated layout.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>, %arg1: tensor<2x4xf32> {tf._layout = "sharding_specs:x,unsharded, mesh:|x=2,y=2|*CPU"}) -> tensor<1x2xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[GATHERED:[0-9]*]] = "tf.DTensorAllGather"(%arg1)
  // CHECK:        "tf.Slice"(%[[GATHERED]], %cst, %cst_1)
  // CHECK-SAME:     _layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
  // CHECK-SAME:     (tensor<2x4xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x2xf32>
  // CHECK:        tf_device.return
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<2x4>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=2|*CPU>} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
    %3 = "tf.Const"() {value = dense<[-1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
    %4 = "tf.Slice"(%1, %2, %3) : (tensor<2x4xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x2xf32>
    tf_device.return %4 : tensor<1x2xf32>
  }) {_mesh = "|x=2,y=2|*CPU", _layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU"]} : () -> (tensor<1x2xf32>)
  func.return %0 : tensor<1x2xf32>
}

// -----

// Check that the slice on sharded x dimension is from 0, which can operate on and produce sharded tensors.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>, %arg1: tensor<2x4xf32> {tf._layout = "sharding_specs:x,unsharded, mesh:|x=2,y=2|*CPU"}) -> tensor<2x2xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:        "tf.Slice"(%arg1, %cst, %cst_1)
  // CHECK-SAME:     _layout = ["sharding_specs:x,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
  // CHECK-SAME:     (tensor<1x4xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x2xf32>
  // CHECK:        tf_device.return
  // CHECK-SAME:     _layout = []
  // CHECK-SAME:     tensor<1x2xf32>
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<2x4>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=2|*CPU>} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = "tf.Const"() {value = dense<[0, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
    %3 = "tf.Const"() {value = dense<[-1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
    %4 = "tf.Slice"(%1, %2, %3) : (tensor<2x4xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xf32>
    tf_device.return %4 : tensor<2x2xf32>
  }) {_mesh = "|x=2,y=2|*CPU", _layout = ["sharding_specs:x,unsharded, mesh:|x=2,y=2|*CPU"]} : () -> (tensor<2x2xf32>)
  func.return %0 : tensor<2x2xf32>
}

// -----

// Slice a sharded input but produce a replicated output. This would crash without the fix for b/181933405.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>, %arg1: tensor<2x4xf32> {tf._layout = "sharding_specs:x,unsharded, mesh:|x=2,y=2|*CPU"}) -> tensor<2x2xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:        "tf.Slice"(%arg1, %cst, %cst_1)
  // CHECK-SAME:     (tensor<1x4xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x2xf32>
  // CHECK:        tf_device.return
  // CHECK-NEXT:     _layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU"], _mesh = "|x=2,y=2|*CPU"
  // CHECK-SAME:     () -> tensor<2x2xf32>
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<2x4>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=2|*CPU>} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = "tf.Const"() {value = dense<[0, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
    %3 = "tf.Const"() {value = dense<[-1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
    %4 = "tf.Slice"(%1, %2, %3) : (tensor<2x4xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xf32>
    %5 = "tf.DTensorLayout"(%4) {global_shape = #tf_type.shape<2x2>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    tf_device.return %5 : tensor<2x2xf32>
  }) {_mesh = "|x=2,y=2|*CPU", _layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU"]} : () -> (tensor<2x2xf32>)
  func.return %0 : tensor<2x2xf32>
}

// -----

// Check SPMD expansion slice with a dynamic begins and sharded input on non
// full slice dimensions.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>,
           %arg1: tensor<2x4xf32> {tf._layout = "sharding_specs:unsharded,x, mesh:|x=2,y=2|*CPU"},
           %arg2: tensor<2xi64> {tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=2|*CPU"}) -> tensor<1x4xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:        %[[SLICE_SIZE:.*]] = "tf.Const"() <{value = dense<[1, 2]> : tensor<2xi64>}> : () -> tensor<2xi64>
  // CHECK-NEXT:   %[[SLICE:.*]] = "tf.Slice"(%arg1, %arg2, %[[SLICE_SIZE]])
  // CHECK-SAME:     _layout = ["sharding_specs:unsharded,x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
  // CHECK-SAME:     (tensor<2x2xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x2xf32>
  // CHECK-NEXT:   tf_device.return
  // CHECK-SAME:     %[[SLICE]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<2x4>, layout = #dtensor.layout<sharding_specs:unsharded,x, mesh:|x=2,y=2|*CPU>} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=2,y=2|*CPU>} : (tensor<2xi64>) -> tensor<2xi64>
    %3 = "tf.Const"() {value = dense<[1, 4]> : tensor<2xi64>} : () -> tensor<2xi64>
    %4 = "tf.Slice"(%1, %2, %3) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x4xf32>
    tf_device.return %4 : tensor<1x4xf32>
  }) {_mesh = "|x=2,y=2|*CPU", _layout = ["sharding_specs:unsharded,x, mesh:|x=2,y=2|*CPU"]} : () -> (tensor<1x4xf32>)
  func.return %0 : tensor<1x4xf32>
}

// -----

// Check SPMD expansion of strided slice op with replicated input.
func.func @main(%arg0: tensor<i32>, %arg1: tensor<2x4xf32> {tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU"}) -> tensor<2x2xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %cst_2 = "tf.Const"() <{value = dense<2> : tensor<2xi32>}>
  // CHECK:        "tf.StridedSlice"(%arg1, %cst, %cst_2, %cst_1)
  // CHECK-SAME:     _layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
  // CHECK-SAME:     (tensor<2x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xf32>
  // CHECK:        tf_device.return
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<2x4>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU>} : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32>
    %3 = "tf.Const"() {value = dense<[2, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
    %4 = "tf.Const"() {value = dense<[1, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %5 = "tf.StridedSlice"(%1, %2, %3, %4) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xf32>
    tf_device.return %5 : tensor<2x2xf32>
  }) {_mesh = "|x=2,y=2|*CPU", _layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU"]} : () -> (tensor<2x2xf32>)
  func.return %0 : tensor<2x2xf32>
}

// -----

// Check layout propagation and spmd expansion of strided slice grad op.
func.func @main(%arg0: tensor<15x12xf32>) -> tensor<15x197x12xf32> {
  // CHECK:      "tf_device.cluster"
  %0 = "tf_device.cluster"() ({
    %cst = "tf.Const"() {_global_shape = [#tf_type.shape<2>], value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>
    %cst_0 = "tf.Const"() {_global_shape = [#tf_type.shape<2>], value = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %cst_1 = "tf.Const"() {_global_shape = [#tf_type.shape<2>], value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32>
    %cst_2 = "tf.Const"() {_global_shape = [#tf_type.shape<3>], value = dense<[15, 197, 12]> : tensor<3xi32>} : () -> tensor<3xi32>
    %1 = "tf.DTensorLayout"(%arg0) {_global_shape = [#tf_type.shape<15x12>], global_shape = #tf_type.shape<15x12>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<15x12xf32>) -> tensor<15x12xf32>
    %2 = "tf.DTensorLayout"(%cst_2) {_global_shape = [#tf_type.shape<3>], global_shape = #tf_type.shape<3>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<3xi32>) -> tensor<3xi32>
    %3 = "tf.DTensorLayout"(%cst_1) {_global_shape = [#tf_type.shape<2>], global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2xi32>) -> tensor<2xi32>
    %4 = "tf.DTensorLayout"(%cst_0) {_global_shape = [#tf_type.shape<2>], global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2xi32>) -> tensor<2xi32>
    %5 = "tf.DTensorLayout"(%cst) {_global_shape = [#tf_type.shape<2>], global_shape = #tf_type.shape<2>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<2xi32>) -> tensor<2xi32>
    %6 = "tf.StridedSliceGrad"(%2, %3, %4, %5, %1) {_global_shape = [#tf_type.shape<15x197x12>], begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 2 : i64} : (tensor<3xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<15x12xf32>) -> tensor<15x197x12xf32>
    %7 = "tf.DTensorLayout"(%6) {_global_shape = [#tf_type.shape<15x197x12>], global_shape = #tf_type.shape<15x197x12>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded,unsharded, mesh:|x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<15x197x12xf32>) -> tensor<15x197x12xf32>
    tf_device.return %7 : tensor<15x197x12xf32>
    }) {_mesh = "|x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"} : () -> tensor<15x197x12xf32>
  func.return %0 : tensor<15x197x12xf32>
}


