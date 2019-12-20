// RUN: tf-opt %s -tf-replicate-to-island | FileCheck %s --dump-input=fail

// Tests per replica island has same control operands as island holding
// replicate.
// CHECK-LABEL: func @controls_per_replica
func @controls_per_replica() {
  tf_executor.graph {
    %1 = tf_executor.ControlTrigger {}
    %2 = tf_executor.ControlTrigger {}
    %3 = tf_executor.island(%1, %2) {
      tf_device.replicate {n = 2 : i32} {
        tf_device.return
      }
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK: %[[CT_0:[0-9]*]] = tf_executor.ControlTrigger
// CHECK: %[[CT_1:[0-9]*]] = tf_executor.ControlTrigger
// CHECK: %[[ISLAND_0:[a-z_0-9]*]] = tf_executor.island(%[[CT_0]], %[[CT_1]])
// CHECK: %[[ISLAND_1:[a-z_0-9]*]] = tf_executor.island(%[[CT_0]], %[[CT_1]])
// CHECK: %[[ISLAND_2:[a-z_0-9]*]] = tf_executor.island(%[[ISLAND_0]], %[[ISLAND_1]])


// Tests devices are not set if no devices were defined in replicate.
// CHECK-LABEL: func @no_devices
func @no_devices() {
  tf_executor.graph {
    %0 = tf_executor.island {
      tf_device.replicate {n = 2 : i32} {
        "tf.opA"() : () -> ()
        tf_device.return
      }
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK:     "tf.opA"
// CHECK-NOT: device
// CHECK:     "tf.opA"
// CHECK-NOT: device


// Tests devices are not set if op already has a device assigned.
// CHECK-LABEL: func @no_override_device
func @no_override_device() {
  tf_executor.graph {
    %0 = tf_executor.island {
      tf_device.replicate {n = 2 : i32, devices = ["/CPU:0", "/GPU:1"]} {
        "tf.opA"() {device = "/TPU:2"} : () -> ()
        tf_device.return
      }
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK:      "tf.opA"
// CHECK-SAME: device = "/TPU:2"
// CHECK:      "tf.opA"
// CHECK-SAME: device = "/TPU:2"


// Tests devices are not set if op is not of the TF dialect.
// CHECK-LABEL: func @no_device_non_tf_ops
func @no_device_non_tf_ops() {
  tf_executor.graph {
    %0 = tf_executor.island {
      tf_device.replicate {n = 2 : i32, devices = ["/CPU:0", "/GPU:1"]} {
        "test.opA"() {device = "/TPU:2"} : () -> ()
        "test.opB"() : () -> ()
        tf_device.return
      }
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK:      "test.opA"
// CHECK-SAME: device = "/TPU:2"
// CHECK:      "test.opB"
// CHECK-NOT:  device
// CHECK:      "test.opA"
// CHECK-SAME: device = "/TPU:2"
// CHECK:      "test.opB"
// CHECK-NOT:  device


// Tests unused per replica island are added as a control dependency to the
// island forwarding per replica results.
// CHECK-LABEL: func @unused_replica_control
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>, %[[ARG_1:[a-z0-9]*]]: tensor<i1>)
func @unused_replica_control(%arg0: tensor<i1>, %arg1: tensor<i1>) {
  %0 = tf_executor.graph {
    %1 = tf_executor.ControlTrigger {}
    %2:2 = tf_executor.island(%1) {
      %3:4 = tf_device.replicate([%arg0, %arg1] as %ri: tensor<i1>) {n = 2 : i32, devices = ["/CPU:0", "/GPU:1"]} {
        %4 = "tf.opA"(%ri) : (tensor<i1>) -> tensor<i1>
        %5 = "tf.opB"(%4) : (tensor<i1>) -> tensor<i1>
        tf_device.return %4, %5 : tensor<i1>, tensor<i1>
      }
      tf_executor.yield %3#0 : tensor<i1>
    }
    tf_executor.fetch %2#0 : tensor<i1>
  }
  return
}

// CHECK:      %[[CT:[0-9]*]] = tf_executor.ControlTrigger
// CHECK:      %[[ISLAND_0:[a-z_0-9]*]]:2, %{{.*}} = tf_executor.island(%[[CT]])
// CHECK:        %[[OP_A_0:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-SAME:   device = "/CPU:0"
// CHECK:        %[[OP_B_0:[0-9]*]] = "tf.opB"(%[[OP_A_0]])
// CHECK-SAME:   device = "/CPU:0"
// CHECK:        tf_executor.yield %[[OP_A_0]], %[[OP_B_0]]
// CHECK:      %[[ISLAND_1:[a-z_0-9]*]]:2, %[[ISLAND_1_control:[a-z_0-9]*]] = tf_executor.island(%[[CT]])
// CHECK:        %[[OP_A_1:[0-9]*]] = "tf.opA"(%[[ARG_1]])
// CHECK-SAME:   device = "/GPU:1"
// CHECK:        %[[OP_B_1:[0-9]*]] = "tf.opB"(%[[OP_A_1]])
// CHECK-SAME:   device = "/GPU:1"
// CHECK:        tf_executor.yield %[[OP_A_1]], %[[OP_B_1]]
// CHECK:      %[[ISLAND_2:.*]], %[[ISLAND_2_control:.*]] = tf_executor.island(%[[ISLAND_1_control]])
// CHECK:        tf_executor.yield %[[ISLAND_0]]#0
// CHECK:      tf_executor.fetch %[[ISLAND_2]]
