// RUN: tf-opt %s -split-input-file -tf-hoist-broadcast-read | FileCheck %s

// The read should be hoisted.

// CHECK-LABEL: func @hoist_cpu
func.func @hoist_cpu(%arg0: tensor<*x!tf_type.resource<tensor<f32>>> {tf.device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"}) -> () {
  // CHECK:      %[[READ:.*]] = "tf.ReadVariableOp"
  // CHECK-NEXT: tf_device.replicate
  // CHECK-NEXT:   "tf.OpA"(%[[READ]])
  tf_device.replicate {n = 2 : i32} {
    %0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    "tf.OpA"(%0) : (tensor<f32>) -> ()
  }
  func.return
}

// -----

// The read should not be hoisted because the resource does not have device type CPU.

// CHECK-LABEL: func @only_hoist_cpu
func.func @only_hoist_cpu(%arg0: tensor<*x!tf_type.resource<tensor<f32>>>) -> () {
  // CHECK:      tf_device.replicate
  // CHECK-NEXT:   "tf.ReadVariableOp"
  tf_device.replicate {n = 2 : i32} {
    %0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    "tf.OpA"(%0) : (tensor<f32>) -> ()
  }
  func.return
}

// -----

// The read should not be hoisted because it follows a write.

// CHECK-LABEL: func @skip_read_after_write
func.func @skip_read_after_write(%arg0: tensor<*x!tf_type.resource<tensor<f32>>> {tf.device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"}) -> () {
  // CHECK:      tf_device.replicate
  // CHECK:        "tf.AssignVariableOp"
  // CHECK-NEXT:   "tf.ReadVariableOp"
  tf_device.replicate {n = 2 : i32} {
    %0 = "tf.OpA"() : () -> tensor<f32>
    "tf.AssignVariableOp"(%arg0, %0) : (tensor<*x!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %1 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    "tf.OpB"(%1) : (tensor<f32>) -> ()
  }
  func.return
}

// -----

// Check that hoisting preserves read order.

// CHECK-LABEL: func @order_preserved
func.func @order_preserved(%arg0: tensor<*x!tf_type.resource<tensor<f32>>> {tf.device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"}, %arg1: tensor<*x!tf_type.resource<tensor<f32>>>, %arg2: tensor<*x!tf_type.resource<tensor<f32>>> {tf.device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"}) -> () {
  // CHECK:      %[[READ0:.*]] = "tf.ReadVariableOp"(%arg0)
  // CHECK-NEXT: %[[READ2:.*]] = "tf.ReadVariableOp"(%arg2)
  // CHECK-NEXT: tf_device.replicate
  // CHECK-NEXT:   %[[READ1:.*]] = "tf.ReadVariableOp"(%arg1)
  // CHECK-NEXT:   "tf.OpA"(%[[READ0]], %[[READ1]], %[[READ2]])
  tf_device.replicate {n = 2 : i32} {
    %0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    %1 = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    %2 = "tf.ReadVariableOp"(%arg2) : (tensor<*x!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    "tf.OpA"(%0, %1, %2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> ()
  }
  func.return
}
