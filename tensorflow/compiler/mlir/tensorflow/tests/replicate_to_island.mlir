// RUN: tf-opt -split-input-file -verify-diagnostics %s -tf-replicate-to-island=legacy-graph-export=false | FileCheck %s

// Tests per replica island has same control operands as island holding
// replicate.
// CHECK-LABEL: func @controls_per_replica
func.func @controls_per_replica() {
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
  func.return
}

// CHECK: %[[CT_0:.*]] = tf_executor.ControlTrigger
// CHECK: %[[CT_1:.*]] = tf_executor.ControlTrigger
// CHECK: %{{.*}} = tf_executor.island(%[[CT_0]], %[[CT_1]])
// CHECK: %{{.*}} = tf_executor.island(%[[CT_0]], %[[CT_1]])


// Tests devices are not remapped if no devices were defined in replicate.
// CHECK-LABEL: func @no_devices
func.func @no_devices() {
  tf_executor.graph {
    %0 = tf_executor.island {
      tf_device.replicate {n = 2 : i32} {
        "tf_device.launch"() ({
          "tf.opA"() : () -> ()
          tf_device.return
        }) {device = "CORE_0"} : () -> ()
        tf_device.return
      }
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}

// CHECK: "tf.opA"
// CHECK: _parallel_execution_ids = "r0:0", device = "CORE_0"
// CHECK: "tf.opA"
// CHECK: _parallel_execution_ids = "r0:1", device = "CORE_0"


// Tests devices are not remapped if device is not in replicate devices.
// CHECK-LABEL: func @no_override_device
func.func @no_override_device() {
  tf_executor.graph {
    %0 = tf_executor.island {
      tf_device.replicate {n = 2 : i32, devices = {CORE_0 = ["/CPU:0", "/GPU:1"]}} {
        "tf_device.launch"() ({
          "tf.opA"() : () -> ()
          tf_device.return
        }) {device = "/TPU:2"} : () -> ()
        tf_device.return
      }
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}

// CHECK: "tf.opA"
// CHECK: _parallel_execution_ids = "r0:0", device = "/TPU:2"
// CHECK: "tf.opA"
// CHECK: _parallel_execution_ids = "r0:1", device = "/TPU:2"


// Tests devices are remapped if device is in replicate devices.
// CHECK-LABEL: func @remap_device
func.func @remap_device() {
  tf_executor.graph {
    %0 = tf_executor.island {
      tf_device.replicate {n = 2 : i32, devices = {CORE_0 = ["/CPU:0", "/GPU:1"]}} {
        "tf_device.launch"() ({
          "tf.opA"() : () -> ()
          tf_device.return
        }) {device = "CORE_0"} : () -> ()
        tf_device.return
      }
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}

// CHECK: "tf.opA"
// CHECK: _parallel_execution_ids = "r0:0", device = "/CPU:0"
// CHECK: "tf.opA"
// CHECK: _parallel_execution_ids = "r0:1", device = "/GPU:1"


// Tests replicate with control dependency output has each expanded replica
// control pinned to a sink island.
// CHECK-LABEL: func @replicate_control
func.func @replicate_control() {
  tf_executor.graph {
    %1 = tf_executor.island {
      tf_device.replicate {n = 2 : i32} {
        tf_device.return
      }
      tf_executor.yield
    }
    tf_executor.fetch %1 : !tf_executor.control
  }
  func.return
}

// CHECK: %[[REPLICA_0:.*]] = tf_executor.island
// CHECK: %[[REPLICA_1:.*]] = tf_executor.island
// CHECK: %[[SINK:.*]] = tf_executor.island(%[[REPLICA_0]], %[[REPLICA_1]])
// CHECK: tf_executor.fetch %[[SINK]]


// Tests unused replica are pinned to the graph fetch.
// CHECK-LABEL: func @unused_replica
func.func @unused_replica(%arg0: tensor<i1>) {
  %0 = tf_executor.graph {
    %1:3 = tf_executor.island {
      %2:2 = tf_device.replicate([%arg0, %arg0] as %ri0: tensor<i1>) {n = 2 : i32} {
        tf_device.return %ri0 : tensor<i1>
      }
      tf_executor.yield %2#0, %2#1 : tensor<i1>, tensor<i1>
    }
    tf_executor.fetch %1#1 : tensor<i1>
  }
  func.return
}

// CHECK: {{%.*}}, [[REPLICA_0_CONTROL:%.*]] = tf_executor.island
// CHECK: [[REPLICA_1_OUTPUT:%.*]], {{%.*}} = tf_executor.island
// CHECK: tf_executor.fetch [[REPLICA_1_OUTPUT]] :


// Tests replicate results are remapped correctly.
// CHECK-LABEL: func @replicate_result
func.func @replicate_result(%arg0: tensor<i1>, %arg1: tensor<i1>) {
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
  func.return
}

// CHECK: %[[REPLICA_OPA_1:.*]], %{{.*}} = tf_executor.island wraps
// CHECK: "tf.opA"(%arg0)
// CHECK:  _parallel_execution_ids = "r0:0"
// CHECK: %[[REPLICA_OPB_1:.*]], %{{.*}} = tf_executor.island wraps
// CHECK: "tf.opB"(%arg0)
// CHECK:  _parallel_execution_ids = "r0:0"
// CHECK: %[[REPLICA_OPA_2:.*]], %{{.*}} = tf_executor.island wraps
// CHECK: "tf.opA"(%arg1)
// CHECK:  _parallel_execution_ids = "r0:1"
// CHECK: %[[REPLICA_OPB_2:.*]], %{{.*}} = tf_executor.island wraps
// CHECK: "tf.opB"(%arg1)
// CHECK:  _parallel_execution_ids = "r0:1"
// CHECK: tf_executor.fetch %[[REPLICA_OPA_1]], %[[REPLICA_OPA_2]], %[[REPLICA_OPB_1]], %[[REPLICA_OPB_2]]

// Tests replicate results are remapped correctly with packed inputs.
// CHECK-LABEL: func @replicate_with_packed_input
func.func @replicate_with_packed_input(%arg0: tensor<i1>, %arg1: tensor<i1>) {
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
  func.return
}

// CHECK: %[[REPLICA_OPA_1:.*]], %{{.*}} = tf_executor.island wraps
// CHECK: "tf.opA"(%arg0)
// CHECK:  _parallel_execution_ids = "r0:0"
// CHECK: %[[REPLICA_OPB_1:.*]], %{{.*}} = tf_executor.island wraps
// CHECK: "tf.opB"(%arg1)
// CHECK:  _parallel_execution_ids = "r0:0"
// CHECK: %[[REPLICA_OPA_2:.*]], %{{.*}} = tf_executor.island wraps
// CHECK: "tf.opA"(%arg0)
// CHECK:  _parallel_execution_ids = "r0:1"
// CHECK: %[[REPLICA_OPB_2:.*]], %{{.*}} = tf_executor.island wraps
// CHECK: "tf.opB"(%arg1)
// CHECK:  _parallel_execution_ids = "r0:1"
// CHECK: tf_executor.fetch %[[REPLICA_OPA_1]], %[[REPLICA_OPA_2]], %[[REPLICA_OPB_1]], %[[REPLICA_OPB_2]]

// Tests replica id is added correctly.
// CHECK-LABEL: func @replica_id_attr_added
func.func @replica_id_attr_added(%arg0: tensor<!tf_type.string>, %arg1: tensor<!tf_type.string>) {
  tf_executor.graph {
    %0 = tf_executor.island {
      tf_device.replicate([%arg0, %arg1] as %arg2: tensor<!tf_type.string>) {n = 2 : i32} {
        "tf.EnqueueTPUEmbeddingSparseTensorBatch"(%arg2){table_ids = [1, 2]} : (tensor<!tf_type.string>) -> ()
        "tf.EnqueueTPUEmbeddingRaggedTensorBatch"(%arg2){table_ids = [1, 2]} : (tensor<!tf_type.string>) -> ()
        "tf.EnqueueTPUEmbeddingArbitraryTensorBatch"(%arg2){table_ids = [1, 2]} : (tensor<!tf_type.string>) -> ()
        "tf.A"(%arg2) : (tensor<!tf_type.string>) -> ()
        tf_device.return
      }
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}

// CHECK:      tf_executor.island wraps
// CHECK:      "tf.EnqueueTPUEmbeddingSparseTensorBatch"
// CHECK-SAME:   _parallel_execution_ids = "r0:0", _xla_replica_id = 0 : i64
// CHECK:      tf_executor.island wraps
// CHECK:      "tf.EnqueueTPUEmbeddingRaggedTensorBatch"
// CHECK-SAME:   _parallel_execution_ids = "r0:0", _xla_replica_id = 0 : i64
// CHECK:      tf_executor.island wraps
// CHECK:      "tf.EnqueueTPUEmbeddingArbitraryTensorBatch"
// CHECK-SAME:   _parallel_execution_ids = "r0:0", _xla_replica_id = 0 : i64
// CHECK:      tf_executor.island wraps "tf.A"
// CHECK-NOT:   _xla_replica_id
// CHECK:      _parallel_execution_ids = "r0:0"
// CHECK:      tf_executor.island wraps
// CHECK:      "tf.EnqueueTPUEmbeddingSparseTensorBatch"
// CHECK-SAME:   _parallel_execution_ids = "r0:1", _xla_replica_id = 1 : i64
// CHECK:      tf_executor.island wraps
// CHECK:      "tf.EnqueueTPUEmbeddingRaggedTensorBatch"
// CHECK-SAME:   _parallel_execution_ids = "r0:1", _xla_replica_id = 1 : i64
// CHECK:      tf_executor.island wraps
// CHECK:      "tf.EnqueueTPUEmbeddingArbitraryTensorBatch"
// CHECK-SAME:   _parallel_execution_ids = "r0:1", _xla_replica_id = 1 : i64
// CHECK:      tf_executor.island wraps "tf.A"
// CHECK-NOT:   _xla_replica_id
// CHECK:      _parallel_execution_ids = "r0:1"
// CHECK:      tf_executor.fetch


// Tests tf._TPUDeviceOrdinalPlaceholder ops are replaced with explicit device
// ordinal constant values based on the first TPU core device id.
// CHECK-LABEL: func @device_ordinals
func.func @device_ordinals() {
  tf_executor.graph {
    %0:3 = tf_executor.island {
      %1:2 = tf_device.replicate {n = 2 : i32, devices = {TPU_REPLICATED_CORE_0 = ["/job:worker/replica:0/task:0/device:TPU:1", "/job:worker/replica:0/task:0/device:TPU:2"]}} {
        %2 = "tf._TPUDeviceOrdinalPlaceholder"() : () -> tensor<i64>
        tf_device.return %2 : tensor<i64>
      }
      tf_executor.yield %1#0, %1#1 : tensor<i64>, tensor<i64>
    }
    tf_executor.fetch
  }
  func.return
}

// CHECK:      tf_executor.island
// CHECK:      [[CONST_0:%.+]] = "tf.Const"
// CHECK-SAME: _parallel_execution_ids = "r0:0", value = dense<1> : tensor<i64>
// CHECK:      tf_executor.yield [[CONST_0]]
// CHECK:      tf_executor.island
// CHECK:      [[CONST_1:%.+]] = "tf.Const"
// CHECK-SAME: _parallel_execution_ids = "r0:1", value = dense<2> : tensor<i64>
// CHECK:      tf_executor.yield [[CONST_1]]

// -----
// Tests parallel_execute nested inside replicate
// CHECK-LABEL: func @nested_parallel_execute
func.func @nested_parallel_execute(%arg0: tensor<i1>, %arg1: tensor<i1>) {
  %0:4 = tf_executor.graph {
    %1:5 = tf_executor.island {
      %2:4 = tf_device.replicate([%arg0, %arg1] as %arg2: tensor<i1>) {n = 2 : i32} {
        %3:2 = "tf_device.parallel_execute"() ({
          %6 = "tf_device.launch"() ({
            %4 = "tf.OpA"(%arg0) : (tensor<i1>) -> tensor<i32>
            tf_device.return %4: tensor<i32>
          }) {device = "/TPU:1"} : () -> (tensor<i32>)
          tf_device.return %6: tensor<i32>
        }, {
          %4 = "tf_device.launch"() ({
            %5 = "tf.OpB"(%arg1) : (tensor<i1>) -> (tensor<i64>)
            tf_device.return %5: tensor<i64>
          }) {device = "/TPU:2"} : () -> (tensor<i64>)
          tf_device.return %4 : tensor<i64>
        }) : () -> (tensor<i32>, tensor<i64>)
        tf_device.return %3#0, %3#1 : tensor<i32>, tensor<i64>
      }
      tf_executor.yield %2#0, %2#1, %2#2, %2#3 : tensor<i32>, tensor<i32>, tensor<i64>, tensor<i64>
    }
    tf_executor.fetch %1#0, %1#1, %1#2, %1#3 : tensor<i32>, tensor<i32>, tensor<i64>, tensor<i64>
  }
  func.return
}

// CHECK:      tf_executor.island
// CHECK:      tf_device.parallel_execute
// CHECK:      tf_device.launch
// CHECK:      tf.OpA
// CHECK:      {device = "/TPU:1"}
// CHECK:      tf_device.launch
// CHECK:      tf.OpB
// CHECK:      {device = "/TPU:2"}
// CHECK:      _parallel_execution_ids = "r0:0"
// CHECK:      tf_executor.island
// CHECK:      tf_device.parallel_execute
// CHECK:      tf_device.launch
// CHECK:      tf.OpA
// CHECK:      {device = "/TPU:1"}
// CHECK:      tf_device.launch
// CHECK:      tf.OpB
// CHECK:      {device = "/TPU:2"}
// CHECK:      _parallel_execution_ids = "r0:1"
// CHECK:      tf_executor.fetch

// -----
// Tests parallel_group attr can merge correctly.
// CHECK-LABEL: func @merge_of_parallel_group_attr
func.func @merge_of_parallel_group_attr() {
  tf_executor.graph {
    %0 = tf_executor.island {
      tf_device.replicate {n = 2 : i32, devices = {CORE_0 = ["/CPU:0", "/GPU:1"]}, _parallel_execution_ids = "r4:5"} {
        "tf_device.launch"() ({
          "tf.opA"() : () -> ()
          tf_device.return
        }) {device = "CORE_0"} : () -> ()
        tf_device.return
      }
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}

// CHECK: "tf.opA"
// CHECK: _parallel_execution_ids = "r4:5,r0:0", device = "/CPU:0"
// CHECK: "tf.opA"
// CHECK: _parallel_execution_ids = "r4:5,r0:1", device = "/GPU:1"

// -----

// Tests tf._TPUDeviceOrdinalPlaceholder cannot be updated when device ordinal
// is missing.

func.func @missing_device_ordinals() {
  tf_executor.graph {
    %0:3 = tf_executor.island {
      %1:2 = tf_device.replicate {n = 2 : i32, devices = {TPU_REPLICATED_CORE_1 = ["/job:worker/replica:0/task:0/device:TPU:1", "/job:worker/replica:0/task:0/device:TPU:2"]}} {
        // expected-error@below {{requires device ordinal from device TPU_REPLICATED_CORE_0 to be present in 'tf.device.replicate' op}}
        %2 = "tf._TPUDeviceOrdinalPlaceholder"() : () -> tensor<i64>
        tf_device.return %2 : tensor<i64>
      }
      tf_executor.yield %1#0, %1#1 : tensor<i64>, tensor<i64>
    }
    tf_executor.fetch
  }
  func.return
}
