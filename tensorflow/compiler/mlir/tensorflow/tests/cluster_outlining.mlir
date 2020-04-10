// RUN: tf-opt %s -split-input-file -tf-device-cluster-outlining | FileCheck %s

// Tests simple case of a single `tf_device.launch`.

module {
  // CHECK-LABEL: func @multiplelaunches
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xi32>)
  func @multiplelaunches(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = tf_executor.graph {
      %1:2 = tf_executor.island {
        // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"(%[[ARG_0]])
        %2 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

        // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf_device.launch_func"(%[[A_OUTPUT]]) {device = "tpu0", func = @tpu0_func}
        %3 = "tf_device.launch"() ( {
          %4 = "tf.B"(%2) : (tensor<?xi32>) -> tensor<?xi32>
          tf_device.return %4 : tensor<?xi32>
        }) {device = "tpu0"} : () -> tensor<?xi32>

        // CHECK: tf_executor.yield %[[C_OUTPUT]]
        tf_executor.yield %3 : tensor<?xi32>
      }
      tf_executor.fetch %1#0 : tensor<?xi32>
    }
    return %0 : tensor<?xi32>
  }

// CHECK-LABEL: func @tpu0_func
// CHECK-SAME: (%[[TPU0_FUNC_ARG_0:[a-z0-9]*]]: tensor<?xi32>) -> tensor<?xi32>
// CHECK-SAME: sym_visibility = "private"
// CHECK: %[[TPU0_FUNC_B_OUTPUT:[0-9]*]] = "tf.B"(%[[TPU0_FUNC_ARG_0]])
// CHECK: return %[[TPU0_FUNC_B_OUTPUT]]
}

// -----

// Tests that multiple `tf_device.launch` that depend on each other are
// correctly handled.

module {
  // CHECK-LABEL: func @multiplelaunches
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xi32>)
  func @multiplelaunches(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = tf_executor.graph {
      %1:2 = tf_executor.island {
        // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"(%[[ARG_0]])
        %2 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

        // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf_device.launch_func"(%[[A_OUTPUT]]) {device = "tpu0", func = @tpu0_func}
        %3 = "tf_device.launch"() ( {
          %6 = "tf.B"(%2) : (tensor<?xi32>) -> tensor<?xi32>
          tf_device.return %6 : tensor<?xi32>
        }) {device = "tpu0"} : () -> tensor<?xi32>

        // CHECK: %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[C_OUTPUT]])
        %4 = "tf.D"(%3) : (tensor<?xi32>) -> tensor<?xi32>

        // CHECK: %[[E_OUTPUT:[0-9]*]] = "tf_device.launch_func"(%[[C_OUTPUT]], %[[D_OUTPUT]]) {device = "gpu0", func = @gpu0_func}
        %5 = "tf_device.launch"() ( {
          %6 = "tf.E"(%3) : (tensor<?xi32>) -> tensor<?xi32>
          %7 = "tf.F"(%4, %6) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
          tf_device.return %7 : tensor<?xi32>
        }) {device = "gpu0"} : () -> tensor<?xi32>

        // CHECK: tf_executor.yield %[[E_OUTPUT]]
        tf_executor.yield %5 : tensor<?xi32>
      }
      tf_executor.fetch %1#0 : tensor<?xi32>
    }
    return %0 : tensor<?xi32>
  }

// CHECK-LABEL: func @tpu0_func
// CHECK-SAME: (%[[TPU0_FUNC_ARG_0:[a-z0-9]*]]: tensor<?xi32>) -> tensor<?xi32>
// CHECK: %[[TPU0_FUNC_B_OUTPUT:[0-9]*]] = "tf.B"(%[[TPU0_FUNC_ARG_0]])
// CHECK: return %[[TPU0_FUNC_B_OUTPUT]]

// CHECK-LABEL: func @gpu0_func
// CHECK-SAME: (%[[GPU0_FUNC_ARG_0:[a-z0-9]*]]: tensor<?xi32>, %[[GPU0_FUNC_ARG_1:[a-z0-9]*]]: tensor<?xi32>) -> tensor<?xi32>
// CHECK: %[[GPU0_FUNC_E_OUTPUT:[0-9]*]] = "tf.E"(%[[GPU0_FUNC_ARG_0]])
// CHECK: %[[GPU0_FUNC_F_OUTPUT:[0-9]*]] = "tf.F"(%[[GPU0_FUNC_ARG_1]], %[[GPU0_FUNC_E_OUTPUT]])
// CHECK: return %[[GPU0_FUNC_F_OUTPUT]]
}

// -----

// Tests outlining launches with no live-in values.

module {
  // CHECK-LABEL: func @multiplelaunches
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xi32>)
  func @multiplelaunches(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = tf_executor.graph {
      %1:2 = tf_executor.island wraps
        // CHECK: %[[A_OUTPUT:[a-z0-9]*]], %{{.*}} = {{.*}} "tf_device.launch_func"() {device = "tpu0", func = @tpu0_func}
        "tf_device.launch"() ( {
          %3 = "tf.A"() : () -> tensor<?xi32>
          tf_device.return %3 : tensor<?xi32>
        }) {device = "tpu0"} : () -> tensor<?xi32>
      // CHECK: tf_executor.fetch %[[A_OUTPUT]]
      tf_executor.fetch %1#0 : tensor<?xi32>
    }
    return %0 : tensor<?xi32>
  }

// CHECK-LABEL: func @tpu0_func
// CHECK-SAME: () -> tensor<?xi32>
// CHECK: %[[TPU0_FUNC_A_OUTPUT:[0-9]*]] = "tf.A"()
// CHECK: return %[[TPU0_FUNC_A_OUTPUT]]
}

// -----

// Tests launch attributes are copied over to launch_func.

module {
  // CHECK-LABEL: func @launch_attrs
  func @launch_attrs() -> tensor<?xi32> {
    %0 = "tf_device.launch"() ( {
      %1 = "tf.A"() : () -> tensor<?xi32>
      tf_device.return %1 : tensor<?xi32>
    }) {device = "tpu0", launch_attr = "launch_attr"} : () -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }

// CHECK: launch_attr = "launch_attr"
}
