 // RUN: dtensor-opt %s -split-input-file -dtensor-multi-device-expansion -verify-diagnostics | FileCheck %s

module attributes {
  tf._default_mesh = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7", tf.devices = {"/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:CPU:1", "/job:localhost/replica:0/task:0/device:CPU:2", "/job:localhost/replica:0/task:0/device:CPU:3", "/job:localhost/replica:0/task:0/device:CPU:4", "/job:localhost/replica:0/task:0/device:CPU:5", "/job:localhost/replica:0/task:0/device:CPU:6", "/job:localhost/replica:0/task:0/device:CPU:7"},
  dtensor.enable_multi_device_mode = true
} {
  func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>}, %arg1: tensor<8xi32> {tf._global_shape = #tf_type.shape<8>, tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7", tf._mesh = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7"}) -> (tensor<8xi32> {tf._global_shape = #tf_type.shape<8>}) attributes {tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0", outputs = "op_output_0"}} {
    %1 = "tf.StatefulPartitionedCall"(%arg0, %arg1) {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7"], _mesh = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7", config = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7", config_proto = "", executor_type = "", f = @_test_func} : (tensor<i32>, tensor<8xi32>) -> tensor<8xi32>
    return %1 : tensor<8xi32>
  }

  func.func private @_test_func(%arg0: tensor<i32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
    return %arg1 : tensor<8xi32>
  }

  // CHECK-LABEL: func.func @main
  // CHECK: %arg0: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}
  // CHECK: %arg1: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:1"}
  // CHECK: %arg2: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:2"}
  // CHECK: %arg3: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:3"}
  // CHECK: %arg4: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:4"}
  // CHECK: %arg5: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:5"}
  // CHECK: %arg6: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:6"}
  // CHECK: %arg7: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:7"}
  // CHECK: tf.entry_function = {inputs = "input_0,input_1,input_2,input_3,input_4,input_5,input_6,input_7", outputs = "output_0,output_1,output_2,output_3,output_4,output_5,output_6,output_7"
  // CHECK: %[[RES:.*]]:8 = "tf.StatefulPartitionedCall"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
  // CHECK-SAME: _layout = ["sharding_specs:unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7"]
  // CHECK-SAME: f = @_multi_device_func_4093838507448400597_971647271862201157
  // CHECK: return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2, %[[RES]]#3, %[[RES]]#4, %[[RES]]#5, %[[RES]]#6, %[[RES]]#7

  // CHECK-LABEL: func.func private @_multi_device_func_4093838507448400597_971647271862201157
  // CHECK: %arg0: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}
  // CHECK: %arg1: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:1"}
  // CHECK: %arg2: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:2"}
  // CHECK: %arg3: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:3"}
  // CHECK: %arg4: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:4"}
  // CHECK: %arg5: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:5"}
  // CHECK: %arg6: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:6"}
  // CHECK: %arg7: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:7"}
  // CHECK: tf.entry_function = {inputs = "input_0,input_1,input_2,input_3,input_4,input_5,input_6,input_7", outputs = "output_0,output_1,output_2,output_3,output_4,output_5,output_6,output_7"
  // CHECK: %[[CST0:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[CST1:.*]] = "tf.Const"
  // CHECK: %[[CST2:.*]] = "tf.Const"
  // CHECK: %[[CST3:.*]] = "tf.Const"
  // CHECK: %[[CST4:.*]] = "tf.Const"
  // CHECK: %[[CST5:.*]] = "tf.Const"
  // CHECK: %[[CST6:.*]] = "tf.Const"
  // CHECK: %[[CST7:.*]] = "tf.Const"
  // CHECK: %[[RES0:.*]] = "tf.StatefulPartitionedCall"(%[[CST0]], %arg0)
  // CHECK: %[[RES1:.*]] = "tf.StatefulPartitionedCall"(%[[CST1]], %arg1)
  // CHECK: %[[RES2:.*]] = "tf.StatefulPartitionedCall"(%[[CST2]], %arg2)
  // CHECK: %[[RES3:.*]] = "tf.StatefulPartitionedCall"(%[[CST3]], %arg3)
  // CHECK: %[[RES4:.*]] = "tf.StatefulPartitionedCall"(%[[CST4]], %arg4)
  // CHECK: %[[RES5:.*]] = "tf.StatefulPartitionedCall"(%[[CST5]], %arg5)
  // CHECK: %[[RES6:.*]] = "tf.StatefulPartitionedCall"(%[[CST6]], %arg6)
  // CHECK: %[[RES7:.*]] = "tf.StatefulPartitionedCall"(%[[CST7]], %arg7)
  // CHECK: return %[[RES0]], %[[RES1]], %[[RES2]], %[[RES3]], %[[RES4]], %[[RES5]], %[[RES6]], %[[RES7]]
}

// -----

// Ensure that expansion doesn't occur when it's disabled.

module attributes { dtensor.enable_multi_device_mode = false } {
  // CHECK-LABEL: func.func @main
  func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>}, %arg1: tensor<8xi32> {tf._global_shape = #tf_type.shape<8>, tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7", tf._mesh = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7"}) -> (tensor<8xi32> {tf._global_shape = #tf_type.shape<8>}) attributes {tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0", outputs = "op_output_0"}} {
    // CHECK: %0 = "tf.StatefulPartitionedCall"(%arg0, %arg1)
    %0 = "tf.StatefulPartitionedCall"(%arg0, %arg1) {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7"], _mesh = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7", config = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7", config_proto = "", executor_type = "", f = @_test_func} : (tensor<i32>, tensor<8xi32>) -> tensor<8xi32>
    return %0 : tensor<8xi32>
  }

  func.func private @_test_func(%arg0: tensor<i32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
    return %arg1 : tensor<8xi32>
  }
}

// -----

// Foo and bar are not valid layouts or meshes, respectively.

module attributes {dtensor.enable_multi_device_mode = true} {
  func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>}, %arg1: tensor<8xi32> {tf._global_shape = #tf_type.shape<8>, tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7", tf._mesh = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7"}) -> (tensor<8xi32> {tf._global_shape = #tf_type.shape<8>}) attributes {tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0", outputs = "op_output_0"}} {
    // expected-error @+1 {{Failed to retrieve op mesh or layout.}}
    %1 = "tf.StatefulPartitionedCall"(%arg0, %arg1) {_layout = ["foo"], _mesh = "bar", config = "", config_proto = "", executor_type = "", f = @_test_func} : (tensor<i32>, tensor<8xi32>) -> tensor<8xi32>
    return %1 : tensor<8xi32>
  }
  func.func private @_test_func(%arg0: tensor<i32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
    return %arg1 : tensor<8xi32>
  }
}

// -----

module attributes {dtensor.enable_multi_device_mode = true} {
  func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>}, %arg1: tensor<8xi32> {tf._global_shape = #tf_type.shape<8>, tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7", tf._mesh = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7"}) -> (tensor<8xi32> {tf._global_shape = #tf_type.shape<8>}) attributes {tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0", outputs = "op_output_0"}} {
    // expected-error @+1 {{Calls must be used by exactly one return op.}}
    %1 = "tf.StatefulPartitionedCall"(%arg0, %arg1) {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7"], _mesh = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7", config = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7", config_proto = "", executor_type = "", f = @_test_func} : (tensor<i32>, tensor<8xi32>) -> tensor<8xi32>
    %2 = "tf.Identity"(%1) : (tensor<8xi32>) -> tensor<8xi32>
    return %2 : tensor<8xi32>
  }
  func.func private @_test_func(%arg0: tensor<i32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
    return %arg1 : tensor<8xi32>
  }
}

// -----

// CHECK-LABEL: module @test_tpu_no_inputs
// CHECK-LABEL: func.func private @_multi_device_func_
// CHECK: %[[RES:.*]] = "tf_device.cluster_func"()
// CHECK: func = @_tpu_func
// CHECK: num_cores_per_replica = 1 : i64
// CHECK: num_replicas = 2 : i64
// CHECK: %[[OUT:.*]]:2 = "tf.TPUPartitionedOutputV2"(%[[RES]])
// CHECK-SAME: _XlaSharding = ""
// CHECK-SAME: partition_dims = []
// CHECK: return %[[OUT]]#0, %[[OUT]]#1

module @test_tpu_no_inputs attributes {dtensor.enable_multi_device_mode = true} {
  func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>}, %arg1: tensor<4xf32> {tf._global_shape = #tf_type.shape<4>, tf._layout = "sharding_specs:unsharded, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", tf._mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"}) -> (tensor<4xf32> {tf._global_shape = #tf_type.shape<4>}) attributes {allow_soft_placement = false, tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0", outputs = "op_output_0"}} {
    %0 = "tf.StatefulPartitionedCall"() {_layout = ["sharding_specs:unsharded, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"], _mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", config = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", config_proto = "", executor_type = "", f = @_cpu_func} : () -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
  func.func private @_cpu_func() -> tensor<4xf32> {
    %0 = "tf_device.cluster_func"() {_tpu_replicate = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", device_assignment = [], func = @_tpu_func, num_cores_per_replica = 1 : i64, padding_map = [], step_marker_location = "", topology = "", use_spmd_for_xla_partitioning = false} : () -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
  func.func private @_tpu_func() -> tensor<4xf32> {
    %cst = "tf.Const"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> tensor<4xf32>
    return %cst : tensor<4xf32>
  }
}

// -----

// CHECK-LABEL: module @test_tpu_with_inputs
// CHECK-LABEL: func.func private @_multi_device_func_
// CHECK: %arg0: tensor<4xf32> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}
// CHECK: %arg1: tensor<4xf32> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:1"}
// CHECK: %[[IN:.*]] = "tf.TPUPartitionedInputV2"(%arg0, %arg1)
// CHECK-SAME: _XlaSharding = ""
// CHECK-SAME: partition_dims = []
// CHECK: %[[RES:.*]] = "tf_device.cluster_func"(%[[IN]])
// CHECK: func = @_tpu_func
// CHECK: num_cores_per_replica = 1 : i64
// CHECK: num_replicas = 2 : i64
// CHECK: %[[OUT:.*]]:2 = "tf.TPUPartitionedOutputV2"(%[[RES]])
// CHECK-SAME: _XlaSharding = ""
// CHECK-SAME: partition_dims = []
// CHECK: return %[[OUT]]#0, %[[OUT]]#1

module @test_tpu_with_inputs attributes {dtensor.enable_multi_device_mode = true} {
  func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>}, %arg1: tensor<4xf32> {tf._global_shape = #tf_type.shape<4>, tf._layout = "sharding_specs:unsharded, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", tf._mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"}) -> (tensor<4xf32> {tf._global_shape = #tf_type.shape<4>}) attributes {allow_soft_placement = false, tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0", outputs = "op_output_0"}} {
    %0 = "tf.StatefulPartitionedCall"(%arg1) {_layout = ["sharding_specs:unsharded, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"], _mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", config = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", config_proto = "", executor_type = "", f = @_cpu_func} : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
  func.func private @_cpu_func(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "tf_device.cluster_func"(%arg0) {_tpu_replicate = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", device_assignment = [], func = @_tpu_func, num_cores_per_replica = 1 : i64, padding_map = [], step_marker_location = "", topology = "", use_spmd_for_xla_partitioning = false} : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
  func.func private @_tpu_func(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }
}

// -----

// CHECK-LABEL: module @test_inferred_resource_attributes
// CHECK-LABEL: func.func @main
// CHECK: "tf.StatefulPartitionedCall"
// CHECK-SAME: _inferred_resource_indices = dense<[1, 2]>
// CHECK-SAME: _inferred_resource_layouts = ["sharding_specs:x,unsharded
// CHECK-SAME , "sharding_specs:unsharded,y

module @test_inferred_resource_attributes attributes {dtensor.all_reduce_combiner.num_ops_in_group = 0 : i64, dtensor.all_reduce_combiner.topological_distance = 0 : i64, dtensor.eager_operation_name = "AssignVariableOp", dtensor.enable_multi_device_mode = true, tf._default_mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1", tf.devices = {"/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:CPU:1"}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1555 : i32}} {
  func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>}, %arg1: tensor<!tf_type.resource<tensor<i32>>> {tf._assigned_resource_local_shape = #tf_type.shape<>, tf._global_shape = #tf_type.shape<>, tf._layout = "empty_layout", tf._mesh = "empty_mesh"}, %arg2: tensor<!tf_type.resource<tensor<i32>>> {tf._assigned_resource_local_shape = #tf_type.shape<>, tf._global_shape = #tf_type.shape<>, tf._layout = "empty_layout", tf._mesh = "empty_mesh"}) attributes {allow_soft_placement = false, tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0,op_input_1", outputs = ""}} {
    "tf.StatefulPartitionedCall"(%arg0, %arg1) {_inferred_resource_indices = dense<1> : vector<1xi32>, _inferred_resource_layouts = ["sharding_specs:x,unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"], _layout = [], _mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1", config = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1", config_proto = "", executor_type = "", f = @_func} : (tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>) -> ()
    "tf.StatefulPartitionedCall"(%arg0, %arg2) {_inferred_resource_indices = dense<2> : vector<1xi32>, _inferred_resource_layouts = ["sharding_specs:unsharded,y, mesh:|x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"], _layout = [], _mesh = "|x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1", config = "|x=1,y=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1", config_proto = "", executor_type = "", f = @_func} : (tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>) -> ()
    return
  }
  func.func private @_func(%arg0: tensor<i32>, %arg1: tensor<!tf_type.resource<tensor<i32>>>) {
    "tf.AssignVariableOp"(%arg1, %arg0) {_global_shape = [], _layout = [], device = "", validate_shape = false} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
    return
  }
}
