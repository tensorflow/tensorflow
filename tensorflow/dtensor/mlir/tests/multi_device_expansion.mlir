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
  // CHECK-SAME: f = @_multi_device_func_5372333290171538790_8525065017853554746
  // CHECK-SAME: _layout = ["sharding_specs:unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7"]
  // CHECK: return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2, %[[RES]]#3, %[[RES]]#4, %[[RES]]#5, %[[RES]]#6, %[[RES]]#7

  // CHECK-LABEL: func.func private @_multi_device_func_5372333290171538790_8525065017853554746(
  // CHECK: %arg0: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}
  // CHECK: %arg1: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:1"}
  // CHECK: %arg2: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:2"}
  // CHECK: %arg3: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:3"}
  // CHECK: %arg4: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:4"}
  // CHECK: %arg5: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:5"}
  // CHECK: %arg6: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:6"}
  // CHECK: %arg7: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:7"}
  // CHECK: tf.entry_function = {inputs = "input_0,input_1,input_2,input_3,input_4,input_5,input_6,input_7", outputs = "output_0,output_1,output_2,output_3,output_4,output_5,output_6,output_7"
  // CHECK: %[[CST0:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
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

// -----

// Tests TPU expansion when the computation returns values.

// CHECK-LABEL: func.func @main
// CHECK-LABEL: func.func private @_func
// CHECK-SAME: %arg0: tensor<1x2xi32> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}
// CHECK-SAME: %arg1: tensor<1x2xi32> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:1"}
// CHECK-SAME: -> (tensor<2xi32>, tensor<2xi32>) {
// CHECK-NEXT:   %0:2 = "tf_device.launch"() <{device = ""}> ({
// CHECK-NEXT:     %compilation_status, %program = "tf._TPUCompileMlir"() <{metadata = ""}> : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
// CHECK-NEXT:     tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<3x!tf_type.string>
// CHECK-NEXT:   }) : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
// CHECK-NEXT:   "tf_device.launch"() <{device = ""}> ({
// CHECK-NEXT:     "tf.TPUCompileSucceededAssert"(%0#0) : (tensor<!tf_type.string>) -> ()
// CHECK-NEXT:     tf_device.return
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT:   %1:2 = "tf_device.parallel_execute"() ({
// CHECK-NEXT:     %2 = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:TPU:0"}> ({
// CHECK-NEXT:       %3 = "tf.TPUExecute"(%arg0, %0#1) : (tensor<1x2xi32>, tensor<3x!tf_type.string>) -> tensor<2xi32>
// CHECK-NEXT:       tf_device.return %3 : tensor<2xi32>
// CHECK-NEXT:     }) : () -> tensor<2xi32>
// CHECK-NEXT:     tf_device.return %2 : tensor<2xi32>
// CHECK-NEXT:   }, {
// CHECK-NEXT:     %2 = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:TPU:1"}> ({
// CHECK-NEXT:       %3 = "tf.TPUExecute"(%arg1, %0#1) : (tensor<1x2xi32>, tensor<3x!tf_type.string>) -> tensor<2xi32>
// CHECK-NEXT:       tf_device.return %3 : tensor<2xi32>
// CHECK-NEXT:     }) : () -> tensor<2xi32>
// CHECK-NEXT:     tf_device.return %2 : tensor<2xi32>
// CHECK-NEXT:   }) : () -> (tensor<2xi32>, tensor<2xi32>)
// CHECK-NEXT:   return %1#0, %1#1 : tensor<2xi32>, tensor<2xi32>
// CHECK-NEXT: }

module attributes {dtensor.all_reduce_combiner.num_ops_in_group = 0 : i64, dtensor.all_reduce_combiner.topological_distance = 0 : i64, dtensor.eager_operation_name = "Sum", dtensor.enable_multi_device_mode = true, tf._default_mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", tf.devices = {"/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:CPU:1", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1625 : i32}} {
  func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>}, %arg1: tensor<1x2xi32> {tf._global_shape = #tf_type.shape<2x2>, tf._layout = "sharding_specs:x,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"}, %arg2: tensor<1xi32> {tf._global_shape = #tf_type.shape<1>, tf._layout = "sharding_specs:unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"}) -> (tensor<2xi32> {tf._global_shape = #tf_type.shape<2>}) attributes {allow_soft_placement = false, tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0,op_input_1", outputs = "op_output_0"}} {
    %0 = "tf.StatefulPartitionedCall"(%arg1) {_layout = ["sharding_specs:unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"], _mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", config = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", config_proto = "", executor_type = "", f = @_func} : (tensor<1x2xi32>) -> tensor<2xi32>
    return %0 : tensor<2xi32>
  }
  func.func private @_func(%arg0: tensor<1x2xi32>) -> tensor<2xi32> {
    %0:2 = "tf_device.launch"() ({
      %compilation_status, %program = "tf._TPUCompileMlir"() {metadata = ""} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
      tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<3x!tf_type.string>
    }) {device = ""} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
    "tf_device.launch"() ({
      "tf.TPUCompileSucceededAssert"(%0#0) : (tensor<!tf_type.string>) -> ()
      tf_device.return
    }) {device = ""} : () -> ()
    %1 = "tf_device.launch"() ({
      %2 = "tf.TPUExecute"(%arg0, %0#1) : (tensor<1x2xi32>, tensor<3x!tf_type.string>) -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }) {device = ""} : () -> tensor<2xi32>
    return %1 : tensor<2xi32>
  }
}

// -----

// Tests TPU expansion when the computation has variable-related operations and does not return values.

// CHECK-LABEL: func.func @main
// CHECK-LABEL: func.func private @_func
// CHECK-SAME: %arg0: tensor<i32> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}
// CHECK-SAME: %arg1: tensor<i32> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:1"}
// CHECK-SAME: %arg2: tensor<!tf_type.resource<tensor<i32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}
// CHECK-SAME: %arg3: tensor<!tf_type.resource<tensor<i32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:1"}
// CHECK-NEXT:   %0:2 = "tf_device.launch"() <{device = ""}> ({
// CHECK-NEXT:     %compilation_status, %program = "tf._TPUCompileMlir"() <{metadata = ""}> : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
// CHECK-NEXT:     tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<3x!tf_type.string>
// CHECK-NEXT:   }) : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
// CHECK-NEXT:   "tf_device.launch"() <{device = ""}> ({
// CHECK-NEXT:     "tf.TPUCompileSucceededAssert"(%0#0) : (tensor<!tf_type.string>) -> ()
// CHECK-NEXT:     tf_device.return
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT:   %1:2 = "tf_device.parallel_execute"() ({
// CHECK-NEXT:     %2 = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:TPU:0"}> ({
// CHECK-NEXT:       %3 = "tf.TPUExecute"(%arg0, %0#1) : (tensor<i32>, tensor<3x!tf_type.string>) -> tensor<i32>
// CHECK-NEXT:       tf_device.return %3 : tensor<i32>
// CHECK-NEXT:     }) : () -> tensor<i32>
// CHECK-NEXT:     tf_device.return %2 : tensor<i32>
// CHECK-NEXT:   }, {
// CHECK-NEXT:     %2 = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:TPU:1"}> ({
// CHECK-NEXT:       %3 = "tf.TPUExecute"(%arg1, %0#1) : (tensor<i32>, tensor<3x!tf_type.string>) -> tensor<i32>
// CHECK-NEXT:       tf_device.return %3 : tensor<i32>
// CHECK-NEXT:     }) : () -> tensor<i32>
// CHECK-NEXT:     tf_device.return %2 : tensor<i32>
// CHECK-NEXT:   }) : () -> (tensor<i32>, tensor<i32>)
// CHECK-NEXT:   "tf.AssignVariableOp"(%arg2, %1#0) <{validate_shape = false}> {_global_shape = [], _layout = [], device = "/job:localhost/replica:0/task:0/device:TPU:0"} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
// CHECK-NEXT:   "tf.AssignVariableOp"(%arg3, %1#1) <{validate_shape = false}> {_global_shape = [], _layout = [], device = "/job:localhost/replica:0/task:0/device:TPU:1"} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
// CHECK-NEXT:   return
// CHECK-NEXT: }

module attributes {dtensor.all_reduce_combiner.num_ops_in_group = 0 : i64, dtensor.all_reduce_combiner.topological_distance = 0 : i64, dtensor.eager_operation_name = "Sum", dtensor.enable_multi_device_mode = true, tf._default_mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", tf.devices = {"/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:CPU:1", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1625 : i32}} {
  func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>}, %arg1: tensor<!tf_type.resource<tensor<i32>>> {tf._assigned_resource_local_shape = #tf_type.shape<>, tf._global_shape = #tf_type.shape<>, tf._layout = "empty_layout", tf._mesh = "empty_mesh"}) -> () attributes {allow_soft_placement = false, tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0", outputs = ""}} {
    "tf.StatefulPartitionedCall"(%arg0, %arg1) {_inferred_resource_indices = dense<1> : vector<1xi32>, _inferred_resource_layouts = ["sharding_specs:x,unsharded, mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"], _layout = [], _mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", config = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", config_proto = "", executor_type = "", f = @_func} : (tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>) -> ()
    return
  }
  func.func private @_func(%arg0: tensor<i32>, %arg1: tensor<!tf_type.resource<tensor<i32>>>) -> () {
    %0:2 = "tf_device.launch"() ({
      %compilation_status, %program = "tf._TPUCompileMlir"() {metadata = ""} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
      tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<3x!tf_type.string>
    }) {device = ""} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
    "tf_device.launch"() ({
      "tf.TPUCompileSucceededAssert"(%0#0) : (tensor<!tf_type.string>) -> ()
      tf_device.return
    }) {device = ""} : () -> ()
    %1 = "tf_device.launch"() ({
      %2 = "tf.TPUExecute"(%arg0, %0#1) : (tensor<i32>, tensor<3x!tf_type.string>) -> tensor<i32>
      tf_device.return %2 : tensor<i32>
    }) {device = ""} : () -> tensor<i32>
    "tf.AssignVariableOp"(%arg1, %1) {_global_shape = [], _layout = [], validate_shape = false} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
    return
  }
}
