// RUN: tf-opt -split-input-file %s -tf-replicate-to-island | FileCheck %s

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


// Tests unused replica are pinned to the graph fetch.
// CHECK-LABEL: func @unused_replica
func @unused_replica(%arg0: tensor<i1>) {
  %0 = tf_executor.graph {
    %1:3 = tf_executor.island {
      %2:2 = tf_device.replicate([%arg0, %arg0] as %ri0: tensor<i1>) {n = 2 : i32} {
        tf_device.return %ri0 : tensor<i1>
      }
      tf_executor.yield %2#0, %2#1 : tensor<i1>, tensor<i1>
    }
    tf_executor.fetch %1#1 : tensor<i1>
  }
  return
}

// CHECK: {{%.*}}, [[REPLICA_0_CONTROL:%.*]] = tf_executor.island
// CHECK: [[REPLICA_1_OUTPUT:%.*]], {{%.*}} = tf_executor.island
// CHECK: tf_executor.fetch [[REPLICA_1_OUTPUT]], [[REPLICA_0_CONTROL]]


// Tests replicate results are remapped correctly.
// CHECK-LABEL: func @replicate_result
func @replicate_result(%arg0: tensor<i1>, %arg1: tensor<i1>) {
  %0:4 = tf_executor.graph {
    %1:5 = tf_executor.island {
      %2:4 = tf_device.replicate([%arg0, %arg1] as %arg2: tensor<i1>) {n = 2 : i32} {
        %3 = "tf.opA"(%arg2) : (tensor<i1>) -> tensor<f32>
        %4 = "tf.opB"(%arg2) : (tensor<i1>) -> tensor<i32>
        tf_device.return %3, %4 : tensor<f32>, tensor<i32>
      }
      tf_executor.yield %2#0, %2#1, %2#2, %2#3 : tensor<f32>, tensor<f32>, tensor<i32>, tensor<i32>
    }
    tf_executor.fetch %1#0, %1#1, %1#2, %1#3 : tensor<f32>, tensor<f32>, tensor<i32>, tensor<i32>
  }
  return
}

// CHECK: %[[REPLICA_0:.*]]:2, %{{.*}} = tf_executor.island
// CHECK: %[[REPLICA_1:.*]]:2, %{{.*}} = tf_executor.island
// CHECK: tf_executor.fetch %[[REPLICA_0]]#0, %[[REPLICA_1]]#0, %[[REPLICA_0]]#1, %[[REPLICA_1]]#1


// Tests replicate results are remapped correctly with packed inputs.
// CHECK-LABEL: func @replicate_with_packed_input
func @replicate_with_packed_input(%arg0: tensor<i1>, %arg1: tensor<i1>) {
  %0:4 = tf_executor.graph {
    %1:5 = tf_executor.island {
      %2:4 = tf_device.replicate(%arg0 as %arg2: tensor<i1>, %arg1 as %arg3: tensor<i1>)
          {n = 2 : i32, _packed_input_indices = [0, 1]} {
        %3 = "tf.opA"(%arg2) : (tensor<i1>) -> tensor<f32>
        %4 = "tf.opB"(%arg3) : (tensor<i1>) -> tensor<i32>
        tf_device.return %3, %4 : tensor<f32>, tensor<i32>
      }
      tf_executor.yield %2#0, %2#1, %2#2, %2#3 : tensor<f32>, tensor<f32>, tensor<i32>, tensor<i32>
    }
    tf_executor.fetch %1#0, %1#1, %1#2, %1#3 : tensor<f32>, tensor<f32>, tensor<i32>, tensor<i32>
  }
  return
}

// CHECK: %[[REPLICA_0:.*]]:2, %{{.*}} = tf_executor.island
// CHECK: "tf.opA"(%arg0)
// CHECK: "tf.opB"(%arg1)
// CHECK: %[[REPLICA_1:.*]]:2, %{{.*}} = tf_executor.island
// CHECK: "tf.opA"(%arg0)
// CHECK: "tf.opB"(%arg1)
// CHECK: tf_executor.fetch %[[REPLICA_0]]#0, %[[REPLICA_1]]#0


// Tests replica id is added correctly.
// CHECK-LABEL: func @replica_id_attr_added
func @replica_id_attr_added(%arg0: tensor<!tf.string>, %arg1: tensor<!tf.string>) {
  tf_executor.graph {
    tf_executor.island {
      tf_device.replicate([%arg0, %arg1] as %arg2: tensor<!tf.string>) {n = 2 : i32} {
        "tf.EnqueueTPUEmbeddingSparseTensorBatch"(%arg2){table_ids = [1, 2]} : (tensor<!tf.string>) -> ()
        "tf.EnqueueTPUEmbeddingRaggedTensorBatch"(%arg2){table_ids = [1, 2]} : (tensor<!tf.string>) -> ()
        "tf.A"(%arg2) : (tensor<!tf.string>) -> ()
        tf_device.return
      }
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK:      tf_executor.island
// CHECK:      "tf.EnqueueTPUEmbeddingSparseTensorBatch"
// CHECK-SAME:   _xla_replica_id = 0
// CHECK:      "tf.EnqueueTPUEmbeddingRaggedTensorBatch"
// CHECK-SAME:   _xla_replica_id = 0
// CHECK:      "tf.A"
// CHECK-NOT:   _xla_replica_id
// CHECK:      tf_executor.island
// CHECK:      "tf.EnqueueTPUEmbeddingSparseTensorBatch"
// CHECK-SAME:   _xla_replica_id = 1
// CHECK:      "tf.EnqueueTPUEmbeddingRaggedTensorBatch"
// CHECK-SAME:   _xla_replica_id = 1
// CHECK:      "tf.A"
// CHECK-NOT:   _xla_replica_id
// CHECK:      tf_executor.fetch


// Tests device ordinals are added to `tf._XlaSendFromHost`/`tf._XlaRecvAtHost`
// based on the first TPU core device id.
// CHECK-LABEL: func @device_ordinals
func @device_ordinals(%arg0: tensor<f32>, %arg1: tensor<2x!tf.string>) {
  tf_executor.graph {
    tf_executor.island {
      tf_device.replicate([%arg0, %arg0] as %arg2: tensor<f32>) {n = 2 : i32, devices = {TPU_REPLICATED_CORE_0 = ["/job:worker/replica:0/task:0/device:TPU:1", "/job:worker/replica:0/task:0/device:TPU:2"]}} {
        %0 = "tf._XlaRecvAtHost"(%arg1) {_xla_has_host_transfer = true, device_ordinal = 0 : i64, key = "host_compute_channel_send_0"} : (tensor<2x!tf.string>) -> tensor<f32>
        "tf._XlaSendFromHost"(%0, %arg1) {_xla_has_host_transfer = true, device_ordinal = 0 : i64, key = "host_compute_channel_recv_0"} : (tensor<f32>, tensor<2x!tf.string>) -> ()
        "tf.NoOp"() : () -> ()
        tf_device.return
      }
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK:      tf_executor.island
// CHECK:      "tf._XlaRecvAtHost"
// CHECK-SAME:   device_ordinal = 1
// CHECK:      "tf._XlaSendFromHost"
// CHECK-SAME:   device_ordinal = 1
// CHECK:      "tf.NoOp"
// CHECK:      tf_executor.island
// CHECK:      "tf._XlaRecvAtHost"
// CHECK-SAME:   device_ordinal = 2
// CHECK:      "tf._XlaSendFromHost"
// CHECK-SAME:   device_ordinal = 2
// CHECK:      "tf.NoOp"

// -----

// Tests functions with replica variant ops reachable from a replicate region
// is cloned and remapped.

// CHECK-LABEL: func @call_with_replicate_variant_ops
func @call_with_replicate_variant_ops(%arg0: tensor<f32>, %arg1: tensor<2x!tf.string>) {
  tf_executor.graph {
    tf_executor.island {
      tf_device.replicate([%arg0, %arg0] as %arg2: tensor<f32>) {n = 2 : i32, devices = {TPU_REPLICATED_CORE_0 = ["/job:worker/replica:0/task:0/device:TPU:1", "/job:worker/replica:0/task:0/device:TPU:2"]}} {
        "tf.StatefulPartitionedCall"(%arg1) {config = "", config_proto = "", executor_type = "", f = @send_recv} : (tensor<2x!tf.string>) -> ()
        tf_device.return
      }
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK: "tf.StatefulPartitionedCall"
// CHECK-SAME: f = [[CALL_REPLICA_0:@[a-z0-9_]+]]
// CHECK: "tf.StatefulPartitionedCall"
// CHECK-SAME: f = [[CALL_REPLICA_1:@[a-z0-9_]+]]

func @send_recv(%arg0: tensor<2x!tf.string>) {
  %0 = "tf._XlaRecvAtHost"(%arg0) {_xla_has_host_transfer = true, device_ordinal = 0 : i64, key = "host_compute_channel_send_0"} : (tensor<2x!tf.string>) -> tensor<f32>
  "tf._XlaSendFromHost"(%0, %arg0) {_xla_has_host_transfer = true, device_ordinal = 0 : i64, key = "host_compute_channel_recv_0"} : (tensor<f32>, tensor<2x!tf.string>) -> ()
  "tf.NoOp"() : () -> ()
  return
}

// CHECK: func [[CALL_REPLICA_0]]
// CHECK: "tf._XlaRecvAtHost"
// CHECK-SAME: device_ordinal = 1
// CHECK: "tf._XlaSendFromHost"
// CHECK-SAME: device_ordinal = 1

// CHECK: func [[CALL_REPLICA_1]]
// CHECK: "tf._XlaRecvAtHost"
// CHECK-SAME: device_ordinal = 2
// CHECK: "tf._XlaSendFromHost"
// CHECK-SAME: device_ordinal = 2

// -----

// Tests transitive functions with replica variant ops reachable from a
// replicate region is cloned and remapped.

// CHECK-LABEL: func @call_with_replicate_variant_ops
func @call_with_replicate_variant_ops(%arg0: tensor<f32>, %arg1: tensor<2x!tf.string>) {
  tf_executor.graph {
    tf_executor.island {
      tf_device.replicate([%arg0, %arg0] as %arg2: tensor<f32>) {n = 2 : i32, devices = {TPU_REPLICATED_CORE_0 = ["/job:worker/replica:0/task:0/device:TPU:1", "/job:worker/replica:0/task:0/device:TPU:2"]}} {
        "tf.StatefulPartitionedCall"(%arg1) {config = "", config_proto = "", executor_type = "", f = @callee} : (tensor<2x!tf.string>) -> ()
        tf_device.return
      }
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK: "tf.StatefulPartitionedCall"
// CHECK-SAME: f = [[CALLEE_REPLICA_0:@[a-z0-9_]+]]
// CHECK: "tf.StatefulPartitionedCall"
// CHECK-SAME: f = [[CALLEE_REPLICA_1:@[a-z0-9_]+]]

func @callee(%arg0: tensor<2x!tf.string>) {
  "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @send_recv} : (tensor<2x!tf.string>) -> ()
  return
}

func @send_recv(%arg0: tensor<2x!tf.string>) {
  %0 = "tf._XlaRecvAtHost"(%arg0) {_xla_has_host_transfer = true, device_ordinal = 0 : i64, key = "host_compute_channel_send_0"} : (tensor<2x!tf.string>) -> tensor<f32>
  "tf._XlaSendFromHost"(%0, %arg0) {_xla_has_host_transfer = true, device_ordinal = 0 : i64, key = "host_compute_channel_recv_0"} : (tensor<f32>, tensor<2x!tf.string>) -> ()
  "tf.NoOp"() : () -> ()
  return
}

// CHECK: func [[CALLEE_REPLICA_0]]
// CHECK: "tf.StatefulPartitionedCall"
// CHECK-SAME: f = [[TRANSITIVE_CALLEE_REPLICA_0:@[a-z0-9_]+]]

// CHECK: func [[TRANSITIVE_CALLEE_REPLICA_0]]
// CHECK: "tf._XlaRecvAtHost"
// CHECK-SAME: device_ordinal = 1
// CHECK: "tf._XlaSendFromHost"
// CHECK-SAME: device_ordinal = 1

// CHECK: func [[CALLEE_REPLICA_1]]
// CHECK: "tf.StatefulPartitionedCall"
// CHECK-SAME: f = [[TRANSITIVE_CALLEE_REPLICA_1:@[a-z0-9_]+]]

// CHECK: func [[TRANSITIVE_CALLEE_REPLICA_1]]
// CHECK: "tf._XlaRecvAtHost"
// CHECK-SAME: device_ordinal = 2
// CHECK: "tf._XlaSendFromHost"
// CHECK-SAME: device_ordinal = 2

// -----

// Tests functional control flow functions with replica variant ops reachable
// from a replicate region is cloned and remapped. Only the branches reachable
// with replica variant ops are cloned.

// CHECK-LABEL: func @control_flow_with_replicate_variant_ops
func @control_flow_with_replicate_variant_ops(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<2x!tf.string>) {
  tf_executor.graph {
    tf_executor.island {
      tf_device.replicate([%arg0, %arg0] as %arg4: tensor<i1>, [%arg1, %arg1] as %arg5: tensor<f32>, [%arg2, %arg2] as %arg6: tensor<f32>) {n = 2 : i32, devices = {TPU_REPLICATED_CORE_0 = ["/job:worker/replica:0/task:0/device:TPU:1", "/job:worker/replica:0/task:0/device:TPU:2"]}} {
        %0 = "tf.If"(%arg4, %arg5, %arg6, %arg3) {else_branch = @cond_false, is_stateless = true, then_branch = @cond_true} : (tensor<i1>, tensor<f32>, tensor<f32>, tensor<2x!tf.string>) -> tensor<f32>
        tf_device.return
      }
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK: "tf.If"
// CHECK-SAME: else_branch = @cond_false
// CHECK-SAME: then_branch = [[COND_TRUE_REPLICA_0:@[a-z0-9_]+]]
// CHECK: "tf.If"
// CHECK-SAME: else_branch = @cond_false
// CHECK-SAME: then_branch = [[COND_TRUE_REPLICA_1:@[a-z0-9_]+]]

func @cond_false(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<2x!tf.string>) -> tensor<f32> {
  return %arg0 : tensor<f32>
}

// CHECK-NOT: func @cond_false.+(

func @cond_true(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<2x!tf.string>) -> tensor<f32> {
  "tf._XlaSendFromHost"(%arg1, %arg2) {_xla_has_host_transfer = true, device_ordinal = 0 : i64, key = "host_compute_channel_recv_0"} : (tensor<f32>, tensor<2x!tf.string>) -> ()
  %0 = "tf._XlaRecvAtHost"(%arg2) {_xla_has_host_transfer = true, device_ordinal = 0 : i64, key = "host_compute_channel_send_0"} : (tensor<2x!tf.string>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK: func [[COND_TRUE_REPLICA_0]]
// CHECK: "tf._XlaSendFromHost"
// CHECK-SAME: device_ordinal = 1
// CHECK: "tf._XlaRecvAtHost"
// CHECK-SAME: device_ordinal = 1

// CHECK: func [[COND_TRUE_REPLICA_1]]
// CHECK: "tf._XlaSendFromHost"
// CHECK-SAME: device_ordinal = 2
// CHECK: "tf._XlaRecvAtHost"
// CHECK-SAME: device_ordinal = 2

// -----

// Tests function with no replica variant ops reachable from a replicate region
// is not cloned.

// CHECK-LABEL: func @no_replicate_variant_ops
func @no_replicate_variant_ops(%arg0: tensor<f32>, %arg1: tensor<2x!tf.string>) {
  tf_executor.graph {
    tf_executor.island {
      tf_device.replicate([%arg0, %arg0] as %arg2: tensor<f32>) {n = 2 : i32, devices = {TPU_REPLICATED_CORE_0 = ["/job:worker/replica:0/task:0/device:TPU:1", "/job:worker/replica:0/task:0/device:TPU:2"]}} {
        "tf.StatefulPartitionedCall"(%arg1) {config = "", config_proto = "", executor_type = "", f = @send_recv} : (tensor<2x!tf.string>) -> ()
        tf_device.return
      }
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK: "tf.StatefulPartitionedCall"
// CHECK-SAME: f = @send_recv

func @send_recv(%arg0: tensor<2x!tf.string>) {
  "tf.NoOp"() : () -> ()
  return
}

// CHECK-NOT: @send_recv.+(
