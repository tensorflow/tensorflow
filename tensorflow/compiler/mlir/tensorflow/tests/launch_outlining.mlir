// RUN: tf-opt %s -split-input-file -tf-device-launch-outlining | FileCheck %s

// Tests simple case of a single `tf_device.launch`.

// CHECK-LABEL: func @single_launch
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xi32>)
func @single_launch(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"(%[[ARG_0]])
      %2 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

      // CHECK: %[[LAUNCH_OUTPUT:[0-9]*]] = "tf_device.launch_func"(%[[A_OUTPUT]]) {device = "/device:test_device:0", func = @[[LAUNCH:.*]]}
      %3 = "tf_device.launch"() ( {
        %4 = "tf.B"(%2) : (tensor<?xi32>) -> tensor<?xi32>
        tf_device.return %4 : tensor<?xi32>
      }) {device = "/device:test_device:0"} : () -> tensor<?xi32>

      // CHECK: tf_executor.yield %[[LAUNCH_OUTPUT]]
      tf_executor.yield %3 : tensor<?xi32>
    }
    tf_executor.fetch %1#0 : tensor<?xi32>
  }
  return %0 : tensor<?xi32>
}

// CHECK: func private @[[LAUNCH]]
// CHECK-SAME: (%[[LAUNCH_ARG_0:[a-z0-9]*]]: tensor<?xi32>) -> tensor<?xi32>
// CHECK: %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[LAUNCH_ARG_0]])
// CHECK: return %[[B_OUTPUT]]

// -----

// Tests that multiple `tf_device.launch` that depend on each other are
// correctly handled.

// CHECK-LABEL: func @multiple_launches
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xi32>)
func @multiple_launches(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"(%[[ARG_0]])
      %2 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

      // CHECK: %[[LAUNCH_0_OUTPUT:[0-9]*]] = "tf_device.launch_func"(%[[A_OUTPUT]]) {device = "/device:test_device:0", func = @[[LAUNCH_0:.*]]}
      %3 = "tf_device.launch"() ( {
        %6 = "tf.B"(%2) : (tensor<?xi32>) -> tensor<?xi32>
        tf_device.return %6 : tensor<?xi32>
      }) {device = "/device:test_device:0"} : () -> tensor<?xi32>

      // CHECK: %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[LAUNCH_0_OUTPUT]])
      %4 = "tf.D"(%3) : (tensor<?xi32>) -> tensor<?xi32>

      // CHECK: %[[LAUNCH_1_OUTPUT:[0-9]*]] = "tf_device.launch_func"(%[[LAUNCH_0_OUTPUT]], %[[D_OUTPUT]]) {device = "/device:test_device:0", func = @[[LAUNCH_1:.*]]}
      %5 = "tf_device.launch"() ( {
        %6 = "tf.E"(%3) : (tensor<?xi32>) -> tensor<?xi32>
        %7 = "tf.F"(%4, %6) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
        tf_device.return %7 : tensor<?xi32>
      }) {device = "/device:test_device:0"} : () -> tensor<?xi32>

      // CHECK: tf_executor.yield %[[LAUNCH_1_OUTPUT]]
      tf_executor.yield %5 : tensor<?xi32>
    }
    tf_executor.fetch %1#0 : tensor<?xi32>
  }
  return %0 : tensor<?xi32>
}

// CHECK: func private @[[LAUNCH_0]]
// CHECK-SAME: (%[[LAUNCH_0_ARG_0:[a-z0-9]*]]: tensor<?xi32>) -> tensor<?xi32>
// CHECK: %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[LAUNCH_0_ARG_0]])
// CHECK: return %[[B_OUTPUT]]

// CHECK: func private @[[LAUNCH_1]]
// CHECK-SAME: (%[[LAUNCH_1_ARG_0:[a-z0-9]*]]: tensor<?xi32>, %[[LAUNCH_1_ARG_1:[a-z0-9]*]]: tensor<?xi32>) -> tensor<?xi32>
// CHECK: %[[E_OUTPUT:[0-9]*]] = "tf.E"(%[[LAUNCH_1_ARG_0]])
// CHECK: %[[F_OUTPUT:[0-9]*]] = "tf.F"(%[[LAUNCH_1_ARG_1]], %[[E_OUTPUT]])
// CHECK: return %[[F_OUTPUT]]

// -----

// Tests outlining launches with no live-in values.

// CHECK-LABEL: func @launch_operands
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xi32>)
func @launch_operands(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island wraps
      // CHECK: %[[LAUNCH_OUTPUT:[a-z0-9]*]], %{{.*}} = {{.*}} "tf_device.launch_func"() {device = "/device:test_device:0", func = @[[LAUNCH:.*]]}
      "tf_device.launch"() ( {
        %3 = "tf.A"() : () -> tensor<?xi32>
        tf_device.return %3 : tensor<?xi32>
      }) {device = "/device:test_device:0"} : () -> tensor<?xi32>
    // CHECK: tf_executor.fetch %[[LAUNCH_OUTPUT]]
    tf_executor.fetch %1#0 : tensor<?xi32>
  }
  return %0 : tensor<?xi32>
}

// CHECK: func private @[[LAUNCH]]
// CHECK-SAME: () -> tensor<?xi32>
// CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"()
// CHECK: return %[[A_OUTPUT]]

// -----

// Tests launch attributes are copied over to launch_func.

// CHECK-LABEL: func @launch_attrs
func @launch_attrs() -> tensor<?xi32> {
  %0 = "tf_device.launch"() ( {
    %1 = "tf.A"() : () -> tensor<?xi32>
    tf_device.return %1 : tensor<?xi32>
  }) {device = "/device:test_device:0", launch_attr = "launch_attr"} : () -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK: "tf_device.launch_func"
// CHECK-SAME: device = "/device:test_device:0"
// CHECK-SAME: launch_attr = "launch_attr"
