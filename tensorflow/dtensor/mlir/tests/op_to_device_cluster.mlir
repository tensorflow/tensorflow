// RUN: dtensor-opt %s -split-input-file -dtensor-op-to-device-cluster -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @check_device_cluster_with_mesh_attribute
func.func @check_device_cluster_with_mesh_attribute() -> tensor<i32> {
  // CHECK:        "tf_device.cluster"
  // CHECK-NEXT:     %[[A_OUT:.*]] = "tf.Const"
  // CHECK-NEXT:     tf_device.return %[[A_OUT]]
  // CHECK-NEXT:   _mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
  %0 = "tf.Const"() {value = dense<10> : tensor<i32>, _mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @check_device_cluster_from_op_mesh
func.func @check_device_cluster_from_op_mesh() -> tensor<i32> {
  // CHECK:        "tf_device.cluster"
  // CHECK-NEXT:     %[[A_OUT:.*]] = "tf.Const"
  // CHECK-NEXT:     tf_device.return %[[A_OUT]]
  // CHECK-NEXT:   _mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
  %0 = "tf.Const"() {value = dense<10> : tensor<i32>, _layout = ["sharding_specs:x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"], _mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @check_device_cluster_from_dtensor_layout_op
func.func @check_device_cluster_from_dtensor_layout_op(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK:        "tf_device.cluster"
  // CHECK-NEXT:     %[[A_OUT:.*]] = "tf.DTensorLayout"
  // CHECK-NEXT:     tf_device.return %[[A_OUT]]
  // CHECK-NEXT:   _mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
  %0 = "tf.DTensorLayout"(%arg0) {global_shape = #tf_type.shape<>, layout = #dtensor.layout<sharding_specs:scalar |x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3>} : (tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @check_device_cluster_from_copy_to_mesh_op
func.func @check_device_cluster_from_copy_to_mesh_op(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK:        "tf_device.cluster"
  // CHECK-NEXT:     %[[A_OUT:.*]] = "tf.Relayout"
  // CHECK-NEXT:     tf_device.return %[[A_OUT]]
  // CHECK-NEXT:   _mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
  %0 = "tf.Relayout"(%arg0) { layout = "sharding_specs:x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : (tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @check_yield_op_ignored
func.func @check_yield_op_ignored(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK:        "tf_device.cluster"
  // CHECK-NEXT:     "tf.WhileRegion"
  // CHECK-NEXT:       bb0(%arg1: tensor<i32>):
  // CHECK-NEXT:       %[[CLUSTER_OUT:.*]] = "tf_device.cluster"
  // CHECK-NEXT:         %[[H_OUT:.*]] = "tf.H"
  // CHECK-NEXT:         tf_device.return %[[H_OUT]]
  // CHECK-NEXT:       () -> tensor<i1>
  // CHECK-NEXT:       "tf.Yield"(%[[CLUSTER_OUT]])
  // CHECK:            ^bb0(%arg1: tensor<i32>):
  // CHECK-NEXT:       "tf.Yield"
  %0 = "tf.WhileRegion"(%arg0) ({
    ^bb0(%arg1: tensor<i32>):
      %1 = "tf.H"(%arg1) :  (tensor<i32>) -> tensor<i1>
      "tf.Yield"(%1) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg1: tensor<i32>):
      "tf.Yield"(%arg1) : (tensor<i32>) -> ()
    }) { is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}
