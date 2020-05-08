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

// CHECK: %[[CT_0:.*]] = tf_executor.ControlTrigger
// CHECK: %[[CT_1:.*]] = tf_executor.ControlTrigger
// CHECK: %{{.*}} = tf_executor.island(%[[CT_0]], %[[CT_1]])
// CHECK: %{{.*}} = tf_executor.island(%[[CT_0]], %[[CT_1]])


// Tests devices are not remapped if no devices were defined in replicate.
// CHECK-LABEL: func @no_devices
func @no_devices() {
  tf_executor.graph {
    %0 = tf_executor.island {
      tf_device.replicate {n = 2 : i32} {
        "tf_device.launch"() ( {
          "tf.opA"() : () -> ()
          tf_device.return
        }) {device = "CORE_0"} : () -> ()
        tf_device.return
      }
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK: "tf.opA"
// CHECK: device = "CORE_0"
// CHECK: "tf.opA"
// CHECK: device = "CORE_0"


// Tests devices are not remapped if device is not in replicate devices.
// CHECK-LABEL: func @no_override_device
func @no_override_device() {
  tf_executor.graph {
    %0 = tf_executor.island {
      tf_device.replicate {n = 2 : i32, devices = {CORE_0 = ["/CPU:0", "/GPU:1"]}} {
        "tf_device.launch"() ( {
          "tf.opA"() : () -> ()
          tf_device.return
        }) {device = "/TPU:2"} : () -> ()
        tf_device.return
      }
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK: "tf.opA"
// CHECK: device = "/TPU:2"
// CHECK: "tf.opA"
// CHECK: device = "/TPU:2"


// Tests devices are remapped if device is in replicate devices.
// CHECK-LABEL: func @remap_device
func @remap_device() {
  tf_executor.graph {
    %0 = tf_executor.island {
      tf_device.replicate {n = 2 : i32, devices = {CORE_0 = ["/CPU:0", "/GPU:1"]}} {
        "tf_device.launch"() ( {
          "tf.opA"() : () -> ()
          tf_device.return
        }) {device = "CORE_0"} : () -> ()
        tf_device.return
      }
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK: "tf.opA"
// CHECK: device = "/CPU:0"
// CHECK: "tf.opA"
// CHECK: device = "/GPU:1"


// Tests replicate with control dependency output has each expanded replica
// control pinned to a sink island.
// CHECK-LABEL: func @replicate_control
func @replicate_control() {
  tf_executor.graph {
    %1 = tf_executor.island {
      tf_device.replicate {n = 2 : i32} {
        tf_device.return
      }
      tf_executor.yield
    }
    tf_executor.fetch %1 : !tf_executor.control
  }
  return
}

// CHECK: %[[REPLICA_0:.*]] = tf_executor.island
// CHECK: %[[REPLICA_1:.*]] = tf_executor.island
// CHECK: %[[SINK:.*]] = tf_executor.island(%[[REPLICA_0]], %[[REPLICA_1]])
// CHECK: tf_executor.fetch %[[SINK]]
