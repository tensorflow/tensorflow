// RUN: dtensor-opt %s -split-input-file -dtensor-allreduce-scatter-optimization -verify-diagnostics | FileCheck %s
//
//

// CHECK-LABEL: func @all_reduce_only
func.func @all_reduce_only() {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const"() {value = dense<0.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %2 = "tf.Const"() {value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    // CHECK:     "tf.DTensorAllReduce"
    // CHECK-NOT: "tf.DTensorReduceScatter"
    %3 = "tf.DTensorAllReduce"(%1, %2) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*TPU"], device_type = "TPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
    "tf_device.return"(%3) : (tensor<4x4xf32>) -> ()
  }) : () -> tensor<4x4xf32>
  "func.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @all_reduce_scatter_2d_major_dim
func.func @all_reduce_scatter_2d_major_dim() {
    // CHECK:               %[[INPUT:.*]] = "tf.Const"() {value = dense<0.0
    // CHECK:               %[[GROUP:.*]] = "tf.Const"() {value =
    // CHECK-SAME{LITERAL}: dense<[[0, 2], [1, 3]]>
    // CHECK:               %[[SCATTER_DIM:.*]] = "tf.Const"() {value = dense<0>
    // CHECK:               "tf.DTensorReduceScatter"(%[[INPUT]], %[[GROUP]], %[[SCATTER_DIM]])
    // CHECK-SAME:          reduce_op = "Add"
    // CHECK-NOT:           "tf.DTensorAllReduce"
    // CHECK-NOT:           "tf.DTensorAllScatter"
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const"() {value = dense<0.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %2 = "tf.Const"() {value = dense<[[0, 2], [1, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %3 = "tf.DTensorAllReduce"(%1, %2) {_layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*TPU"], device_type = "TPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
    %4 = "tf.DTensorAllScatter"(%3) {_layout = ["sharding_specs:x,unsharded, mesh:|x=2,y=2|*TPU"], input_layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>, output_layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<4x4xf32>) -> tensor<2x4xf32>
    "tf_device.return"(%4) : (tensor<2x4xf32>) -> ()
  }) : () -> tensor<2x4xf32>
  "func.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @all_reduce_scatter_2d_minor_dim
func.func @all_reduce_scatter_2d_minor_dim() {
    // CHECK:               %[[INPUT:.*]] = "tf.Const"() {value = dense<0.0
    // CHECK:               %[[GROUP:.*]] = "tf.Const"() {value =
    // CHECK-SAME{LITERAL}: dense<[[0, 2], [1, 3]]>
    // CHECK:               %[[SCATTER_DIM:.*]] = "tf.Const"() {value = dense<1>
    // CHECK:               "tf.DTensorReduceScatter"(%[[INPUT]], %[[GROUP]], %[[SCATTER_DIM]])
    // CHECK-SAME:          reduce_op = "Add"
    // CHECK-NOT:           "tf.DTensorAllReduce"
    // CHECK-NOT:           "tf.DTensorAllScatter"
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const"() {value = dense<0.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %2 = "tf.Const"() {value = dense<[[0, 2], [1, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %3 = "tf.DTensorAllReduce"(%1, %2) {_layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*TPU"], device_type = "TPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
    %4 = "tf.DTensorAllScatter"(%3) {_layout = ["sharding_specs:unsharded,x, mesh:|x=2,y=2|*TPU"], input_layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>, output_layout = #dtensor.layout<sharding_specs:unsharded,x, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<4x4xf32>) -> tensor<4x2xf32>
    "tf_device.return"(%4) : (tensor<4x2xf32>) -> ()
  }) : () -> tensor<4x2xf32>
  "func.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @all_reduce_multiple_users
func.func @all_reduce_multiple_users() {
    // CHECK:     "tf.DTensorAllReduce"
    // CHECK:     "tf.DTensorAllScatter"
    // CHECK-NOT: "tf.DTensorReduceScatter"
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const"() {value = dense<0.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %2 = "tf.Const"() {value = dense<[[0, 2], [1, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %3 = "tf.DTensorAllReduce"(%1, %2) {_layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*TPU"], device_type = "TPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
    %4 = "tf.DTensorAllScatter"(%3) {_layout = ["sharding_specs:x,unsharded, mesh:|x=2,y=2|*TPU"], input_layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>, output_layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<4x4xf32>) -> tensor<2x4xf32>
    %5 = "tf.Identity"(%3) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    "tf_device.return"(%4) : (tensor<2x4xf32>) -> ()
  }) : () -> tensor<2x4xf32>
  "func.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @all_reduce_scatter_2d_mismatched_dim
func.func @all_reduce_scatter_2d_mismatched_dim() {
  %0 = "tf_device.cluster"() ({
    // CHECK:     "tf.DTensorAllReduce"
    // CHECK:     "tf.DTensorAllScatter"
    // CHECK-NOT: "tf.DTensorReduceScatter"
    %1 = "tf.Const"() {value = dense<0.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %2 = "tf.Const"() {value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %3 = "tf.DTensorAllReduce"(%1, %2) {_layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*TPU"], device_type = "TPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
    %4 = "tf.DTensorAllScatter"(%3) {_layout = ["sharding_specs:x,unsharded, mesh:|x=2,y=2|*TPU"], input_layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>, output_layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<4x4xf32>) -> tensor<2x4xf32>
    "tf_device.return"(%4) : (tensor<2x4xf32>) -> ()
  }) : () -> tensor<2x4xf32>
  "func.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @all_reduce_scatter_2d_both_dims
func.func @all_reduce_scatter_2d_both_dims() {
    // CHECK:     "tf.DTensorAllReduce"
    // CHECK:     "tf.DTensorAllScatter"
    // CHECK-NOT: "tf.DTensorReduceScatter"
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const"() {value = dense<0.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %2 = "tf.Const"() {value = dense<[[0, 2], [1, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %3 = "tf.DTensorAllReduce"(%1, %2) {_layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*TPU"], device_type = "TPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
    %4 = "tf.DTensorAllScatter"(%3) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*TPU"], input_layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>, output_layout = #dtensor.layout<sharding_specs:x,y, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3>} : (tensor<4x4xf32>) -> tensor<2x2xf32>
    "tf_device.return"(%4) : (tensor<2x2xf32>) -> ()
  }) : () -> tensor<2x2xf32>
  "func.return"() : () -> ()
}
