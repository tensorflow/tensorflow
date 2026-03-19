// RUN: dtensor-opt %s -split-input-file -dtensor-annotate-global-shape -dtensor-spmd-expansion -verify-diagnostics | FileCheck %s

// UnsortedSegmentSum data and segment sum same layout.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1xi32>,
           %arg1: tensor<16x2xf32> {tf._layout = "sharding_specs:x,unsharded, mesh:TPU|x=4|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"},
           %arg2: tensor<16xi32> {tf._layout = "sharding_specs:x, mesh:TPU|x=4|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"}) {
  // CHECK:    "tf_device.cluster"
  // CHECK:      %[[NUM_SEGMENTS:.*]] = "tf.Const"
  // CHECK-SAME: () -> tensor<i32>
  // CHECK:      %[[LOCAL_RESULT:.*]] = "tf.UnsortedSegmentSum"(%arg1, %arg2, %[[NUM_SEGMENTS]])
  // CHECK-SAME: (tensor<4x2xf32>, tensor<4xi32>, tensor<i32>) -> tensor<8x2xf32>
  // CHECK:      %[[RESULT:.*]] = "tf.DTensorAllReduce"(%[[LOCAL_RESULT]]
  // CHECK-SAME: reduce_op = "Add"
  // CHECK-SAME: _layout = ["sharding_specs:unsharded,unsharded, mesh:TPU|x=4|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
  // CHECK:      %[[FINAL_RESULT:.*]] = "tf.DTensorAllScatter"(%[[RESULT]]
  // CHECK-NEXT: tf_device.return
  // CHECK-SAME: %[[FINAL_RESULT]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<16x2>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:TPU|x=4|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<16x2xf32>) -> tensor<16x2xf32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<16>, layout = #dtensor.layout<sharding_specs:x, mesh:TPU|x=4|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<16xi32>) -> tensor<16xi32>
    %3 = "tf.Const"() {value = dense<8> : tensor<i32>} : () -> tensor<i32>
    %4 = "tf.DTensorLayout"(%3) {global_shape = #tf_type.shape<>, layout = #dtensor.layout<sharding_specs: mesh:TPU|x=4|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<i32>) -> tensor<i32>
    %5 = "tf.UnsortedSegmentSum"(%1, %2, %4) : (tensor<16x2xf32>, tensor<16xi32>, tensor<i32>) -> tensor<8x2xf32>
    %6 = "tf.DTensorLayout"(%5) {global_shape = #tf_type.shape<8x2>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:TPU|x=4|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<8x2xf32>) -> tensor<8x2xf32>
    tf_device.return %6 : tensor<8x2xf32>
  }) {_mesh = "TPU|x=4|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> (tensor<i32>)
  func.return
}
