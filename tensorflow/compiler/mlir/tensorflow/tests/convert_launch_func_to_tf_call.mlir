// RUN: tf-opt %s -split-input-file -tf-device-convert-launch-func-to-tf-call | FileCheck %s

// Tests a single `tf_device.launch_func`.

// CHECK-LABEL: func @single_launch_func
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xf32>)
func @single_launch_func(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"(%[[ARG_0]])
      %2 = "tf.A"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>

      // CHECK: %[[CALL_OUTPUT:[0-9]*]] = "tf.PartitionedCall"(%[[A_OUTPUT]])
      // CHECK-SAME: device = "/device:test_device:0"
      // CHECK-SAME: f = @_func
      %3 = "tf_device.launch_func"(%2) {device = "/device:test_device:0", func = @_func} : (tensor<?xf32>) -> tensor<?xf32>

      // CHECK: tf_executor.yield %[[CALL_OUTPUT]]
      tf_executor.yield %3 : tensor<?xf32>
    }
    tf_executor.fetch %1#0 : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}

func @_func(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  return %arg0 : tensor<?xf32>
}

// -----

// Tests multiple `tf_device.launch_func`.

// CHECK-LABEL: func @multi_launch_func
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xf32>)
func @multi_launch_func(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"(%[[ARG_0]])
      %2 = "tf.A"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>

      // CHECK: %[[CALL_OUTPUT_0:[0-9]*]] = "tf.PartitionedCall"(%[[A_OUTPUT]])
      // CHECK-SAME: device = "/device:test_device:0"
      // CHECK-SAME: f = @_func
      %3 = "tf_device.launch_func"(%2) {device = "/device:test_device:0", func = @_func} : (tensor<?xf32>) -> tensor<?xf32>

      // CHECK: %[[CALL_OUTPUT_1:[0-9]*]] = "tf.PartitionedCall"(%[[CALL_OUTPUT_0]])
      // CHECK-SAME: device = "/device:test_device:1"
      // CHECK-SAME: f = @_func
      %4 = "tf_device.launch_func"(%3) {device = "/device:test_device:1", func = @_func} : (tensor<?xf32>) -> tensor<?xf32>

      // CHECK: tf_executor.yield %[[CALL_OUTPUT_1]]
      tf_executor.yield %4 : tensor<?xf32>
    }
    tf_executor.fetch %1#0 : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}

func @_func(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  return %arg0 : tensor<?xf32>
}
