// RUN: dtensor-opt %s -split-input-file -dtensor-multi-device-expansion -verify-diagnostics | FileCheck %s

module attributes {tf._default_mesh = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7", tf.devices = {"/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:CPU:1", "/job:localhost/replica:0/task:0/device:CPU:2", "/job:localhost/replica:0/task:0/device:CPU:3", "/job:localhost/replica:0/task:0/device:CPU:4", "/job:localhost/replica:0/task:0/device:CPU:5", "/job:localhost/replica:0/task:0/device:CPU:6", "/job:localhost/replica:0/task:0/device:CPU:7"}} {
  // CHECK-LABEL: func @main
  // CHECK: %arg0: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}
  // CHECK: %arg1: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:1"}
  // CHECK: %arg2: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:2"}
  // CHECK: %arg3: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:3"}
  // CHECK: %arg4: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:4"}
  // CHECK: %arg5: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:5"}
  // CHECK: %arg6: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:6"}
  // CHECK: %arg7: tensor<8xi32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:7"}
  func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>}, %arg1: tensor<8xi32> {tf._global_shape = #tf_type.shape<8>, tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7", tf._mesh = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7"}) -> (tensor<8xi32> {tf._global_shape = #tf_type.shape<8>}) attributes {tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0", outputs = "op_output_0"}} {
    // CHECK: %[[CST0:.*]] = "tf.Const"
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
    %1 = "tf.StatefulPartitionedCall"(%arg0, %arg1) {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7"], _mesh = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7", config = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7", config_proto = "", executor_type = "", f = @_test_func} : (tensor<i32>, tensor<8xi32>) -> tensor<8xi32>
    // CHECK: return %[[RES0]], %[[RES1]], %[[RES2]], %[[RES3]], %[[RES4]], %[[RES5]], %[[RES6]], %[[RES7]]
    return %1 : tensor<8xi32>
  }
  func.func private @_test_func(%arg0: tensor<i32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
    return %arg1 : tensor<8xi32>
  }
}

// -----

// Foo and bar are not valid layouts or meshes, respectively.

module {
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

module {
  func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>}, %arg1: tensor<8xi32> {tf._global_shape = #tf_type.shape<8>, tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7", tf._mesh = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7"}) -> (tensor<8xi32> {tf._global_shape = #tf_type.shape<8>}) attributes {tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0", outputs = "op_output_0"}} {
    // expected-error @+1 {{Call result must be used by return op.}}
    %1 = "tf.StatefulPartitionedCall"(%arg0, %arg1) {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7"], _mesh = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7", config = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3,/job:localhost/replica:0/task:0/device:CPU:4,/job:localhost/replica:0/task:0/device:CPU:5,/job:localhost/replica:0/task:0/device:CPU:6,/job:localhost/replica:0/task:0/device:CPU:7", config_proto = "", executor_type = "", f = @_test_func} : (tensor<i32>, tensor<8xi32>) -> tensor<8xi32>
    %2 = "tf.Identity"(%1) : (tensor<8xi32>) -> tensor<8xi32>
    return %2 : tensor<8xi32>
  }
  func.func private @_test_func(%arg0: tensor<i32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
    return %arg1 : tensor<8xi32>
  }
}
