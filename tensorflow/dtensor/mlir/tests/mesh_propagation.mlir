// RUN: dtensor-opt %s -split-input-file -dtensor-mesh-propagation -verify-diagnostics | FileCheck %s

// Checks that default mesh is propagated.
// CHECK-LABEL: module @test_default_mesh
// CHECK-SAME: tf._default_mesh = "[[DEFAULT_MESH:.*]]"
module @test_default_mesh attributes {tf._default_mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3" } {
  // CHECK: func.func @main
  func.func @main(%arg0: tensor<i32>) -> tensor<2xi32> {
    // CHECK: "tf_device.cluster"
    // CHECK-NEXT: "tf.Const"
    // CHECK-NEXT: tf_device.return
    // CHECK-NEXT: _mesh = "[[DEFAULT_MESH]]"
    %0 = "tf_device.cluster"() ({
      %cst = "tf.Const"() {device = "", value = dense<[3, 4]> : tensor<2xi32>} : () -> tensor<2xi32>
      tf_device.return %cst : tensor<2xi32>
    }) : () -> tensor<2xi32>
    // CHECK: "tf_device.cluster"
    // CHECK-NEXT: "tf.Identity"
    // CHECK-NEXT: tf_device.return
    // CHECK-NEXT: _mesh = "[[DEFAULT_MESH]]"
    %1 = "tf_device.cluster"() ({
      %2 = "tf.Identity"(%0) {device = ""} : (tensor<2xi32>) -> tensor<2xi32>
      tf_device.return %2 : tensor<2xi32>
    }) : () -> tensor<2xi32>
    return %1 : tensor<2xi32>
  }
}

// -----

// Checks that input mesh is correctly propagated to it's consumers.
// CHECK-LABEL: module @test_input_mesh
module @test_input_mesh {
  func.func @main() {
    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:      %[[A_OUT:.*]] = "tf.A"
    // CHECK-NEXT:      tf_device.return %[[A_OUT]]
    // CHECK-NEXT:    _mesh = "CPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"
    %0 = "tf_device.cluster"() ({
      %1 = "tf.A"() : () -> tensor<i32>
      tf_device.return %1 : tensor<i32>
    }) {_mesh = "CPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"} : () -> (tensor<i32>)

    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:      %[[B_OUT:.*]] = "tf.B"
    // CHECK-NEXT:      tf_device.return %[[B_OUT]]
    // CHECK-NEXT:    _mesh = "CPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"
    %2 = "tf_device.cluster"() ({
      %3 = "tf.B"(%0) : (tensor<i32>) -> tensor<f32>
      tf_device.return %3 : tensor<f32>
    }) : () -> (tensor<f32>)
    func.return
  }
}

// -----

// Checks that mesh is propagated from inputs of `tf_device.Cluster` op if the
// inputs are arguments of enclosing function.
// CHECK-LABEL: module @test_args_of_enclosing_func
module @test_args_of_enclosing_func {
  func.func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32> {tf._mesh = "TPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"}, %arg2: tensor<1xf32> {tf._mesh = "TPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"}) -> () {
    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:      %[[A_OUT:.*]] = "tf.A"
    // CHECK-NEXT:      tf_device.return
    // CHECK-NEXT:    _mesh = "TPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"
    %0 = "tf_device.cluster"() ({
      %1 = "tf.A"(%arg1, %arg2) : (tensor<1xf32>, tensor<1xf32>) -> tensor<i32>
      tf_device.return %1 : tensor<i32>
    }) : () -> (tensor<i32>)
    func.return
  }
}

// -----


func.func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32> {tf._mesh = "CPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"}, %arg2: tensor<1xf32> {tf._mesh = "TPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"}) -> () {
    // expected-error @+1 {{ All inputs to `tf_device.Cluster` must have same mesh configuration}}
    %0 = "tf_device.cluster"() ({
      %1 = "tf.A"(%arg1, %arg2) : (tensor<1xf32>, tensor<1xf32>) -> tensor<i32>
      tf_device.return %1 : tensor<i32>
    }) : () -> (tensor<i32>)
    func.return
}

// -----

// Checks that mesh is correctly propagated from `tf_device.Cluster` op's consumers.
// CHECK-LABEL: module @test_cluster_to_consumers
module @test_cluster_to_consumers {
  func.func @main() {
    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:      %[[A_OUT:.*]] = "tf.A"
    // CHECK-NEXT:      tf_device.return %[[A_OUT]]
    // CHECK-NEXT:    _mesh = "CPU|batch=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"
    %0 = "tf_device.cluster"() ({
      %1 = "tf.A"() : () -> tensor<i32>
      tf_device.return %1 : tensor<i32>
    }) : () -> (tensor<i32>)

    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:      %[[B_OUT:.*]] = "tf.B"
    // CHECK-NEXT:      tf_device.return %[[B_OUT]]
    // CHECK-NEXT:    _mesh = "CPU|batch=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"
    %2 = "tf_device.cluster"() ({
      %3 = "tf.B"(%0) : (tensor<i32>) -> tensor<f32>
      tf_device.return %3 : tensor<f32>
    }) {_mesh = "CPU|batch=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"} : () -> (tensor<f32>)
    func.return
  }
}

// -----

// Checks that mesh is correctly propagated from default layout of the enclosing function.
// CHECK-LABEL: module @test_default_layout
module @test_default_layout {
  func.func @main() ->(tensor<i32>{tf._default_layout = "sharding_specs:unsharded, mesh:CPU|batch=2,x=1|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"}) {
    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:      %[[A_OUT:.*]] = "tf.A"
    // CHECK-NEXT:      tf_device.return %[[A_OUT]]
    // CHECK-NEXT:   _mesh = "CPU|batch=2,x=1|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"
    %0 = "tf_device.cluster"() ({
      %1 = "tf.A"() : () -> tensor<i32>
      tf_device.return %1 : tensor<i32>
    }) : () -> (tensor<i32>)
    func.return %0 : tensor<i32>
  }
}

// -----

// Checks that mesh is propagated from function arguments and operands for
// nested function.
// CHECK-LABEL: module @test_nested_func_args
module @test_nested_func_args {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<!tf_type.resource> {
    tf._layout = "sharding_specs:unsharded, mesh:CPU|batch=1,x=1|0|0|/job:localhost/task:0/device:CPU:0",
    tf._mesh = "CPU|batch=1,x=1|0|0|/job:localhost/task:0/device:CPU:0"}) ->
  tensor<2xi64> attributes {tf.entry_function = {control_outputs = "", inputs = "device_id,op_input_0", outputs = "op_output_0"}} {
    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:      %[[A_OUT:.*]] = "tf.A"
    // CHECK-NEXT:      tf_device.return %[[A_OUT]]
    // CHECK-NEXT:   _mesh = "CPU|batch=1,x=1|0|0|/job:localhost/task:0/device:CPU:0"
    %0 = "tf_device.cluster"() ({
      %1= "tf.A"(%arg1) : (tensor<!tf_type.resource>) -> (tensor<2xi64>)
      tf_device.return %1 : tensor<2xi64>
    }) : () -> tensor<2xi64>

    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:      %[[CALL_OUT:.*]] = "tf.PartitionedCall"
    // CHECK-NEXT:      tf_device.return %[[CALL_OUT]]
    // CHECK-NEXT:   _mesh = "CPU|batch=1,x=1|0|0|/job:localhost/task:0/device:CPU:0"
    %1 = "tf_device.cluster"() ({
      %2 = "tf.PartitionedCall"(%0, %arg0) {f = @callee, config = "", config_proto = "", executor_type = ""} : (tensor<2xi64>, tensor<i32>) -> (tensor<2xi64>)
      tf_device.return %2 : tensor<2xi64>
    }) : () -> tensor<2xi64>
    func.return %1 : tensor<2xi64>
  }

  // CHECK: func private @callee
  // CHECK-SAME: %arg0: tensor<2xi64>
  // CHECK-SAME: tf._mesh = "CPU|batch=1,x=1|0|0|/job:localhost/task:0/device:CPU:0"
  // CHECK-SAME: %arg1: tensor<i32>
  func.func private @callee(%arg0: tensor<2xi64>, %arg1: tensor<i32>) -> tensor<2xi64> attributes {tf.signature.is_stateful} {
    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:      %[[B_OUT:.*]] = "tf.B"
    // CHECK-NEXT:      tf_device.return %[[B_OUT]]
    // CHECK-NEXT:   _mesh = "CPU|batch=1,x=1|0|0|/job:localhost/task:0/device:CPU:0"
    %1 = "tf_device.cluster"() ({
      %0 = "tf.B"(%arg0, %arg1) : (tensor<2xi64>, tensor<i32>) -> (tensor<2xi64>)
      tf_device.return %0 : tensor<2xi64>
    }) : () -> tensor<2xi64>
    func.return %1 : tensor<2xi64>
  }
}

// -----

// Checks that mesh is propagated for functions without outputs from functions'
// arguments.
// CHECK-LABEL: module @test_no_outputs
module @test_no_outputs {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<!tf_type.resource> {
    tf._layout = "sharding_specs:unsharded, CPU|batch=1,x=1|0|0|/job:localhost/task:0/device:CPU:0",
    tf._mesh = "CPU|batch=1,x=1|0|0|/job:localhost/task:0/device:CPU:0"}) ->
  () attributes {tf.entry_function = {control_outputs = "", inputs = "device_id,op_input_0", outputs = ""}} {
    // CHECK:   "tf_device.cluster"
    // CHECK:   _mesh = "CPU|batch=1,x=1|0|0|/job:localhost/task:0/device:CPU:0"
    "tf_device.cluster"() ({
      "tf.PartitionedCall"(%arg0, %arg1) {f = @assign_var, config = "", config_proto = "", executor_type = ""} : (tensor<i32>, tensor<!tf_type.resource>) -> ()
      tf_device.return
    }) : () -> ()
    func.return
  }

  // CHECK: func private @assign_var
  // CHECK-SAME: %arg0: tensor<i32>
  // CHECK-SAME: %arg1: tensor<!tf_type.resource>
  func.func private @assign_var(%arg0: tensor<i32>, %arg1: tensor<!tf_type.resource>) -> () attributes {tf.signature.is_stateful} {
    // CHECK:   "tf_device.cluster"
    // CHECK:   _mesh = "CPU|batch=1,x=1|0|0|/job:localhost/task:0/device:CPU:0"
    "tf_device.cluster"() ({
      "tf.A"(%arg0, %arg1) : (tensor<i32>, tensor<!tf_type.resource2>) -> ()
      tf_device.return
    }) : () -> ()
    func.return
  }
}

// -----

// Checks that mesh is propagated from consumers for nested functions.
// CHECK-LABEL: module @test_nested_func_ret
module @test_nested_func_ret {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<!tf_type.resource>) -> (tensor<2xi64>{tf._default_layout ="sharding_specs:unsharded, CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"})
    attributes {tf.entry_function = {control_outputs = "", inputs = "device_id,op_input_0", outputs = "op_output_0"}} {
    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:      %[[A_OUT:.*]] = "tf.A"
    // CHECK-NEXT:      tf_device.return %[[A_OUT]]
    // CHECK-NEXT:   _mesh = "CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
    %0 = "tf_device.cluster"() ({
      %1= "tf.A"(%arg1) : (tensor<!tf_type.resource>) -> (tensor<2xi64>)
      tf_device.return %1 : tensor<2xi64>
    }) : () -> tensor<2xi64>

    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:      %[[CALL_OUT:.*]]:2 = "tf.PartitionedCall"
    // CHECK-NEXT:      tf_device.return %[[CALL_OUT]]#0
    // CHECK:         _mesh = "CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
    %1 = "tf_device.cluster"() ({
      %2, %3 = "tf.PartitionedCall"(%0, %arg0) {f = @callee, config = "", config_proto = "", executor_type = ""} : (tensor<2xi64>, tensor<i32>) -> (tensor<2xi64>, tensor<i32>)
      tf_device.return %2 : tensor<2xi64>
    }) : () -> tensor<2xi64>
    func.return %1 : tensor<2xi64>
  }

  // CHECK: func private @callee
  // CHECK-SAME:  %arg0: tensor<2xi64>
  // CHECK-SAME:  %arg1: tensor<i32>
  // CHECK:       tf._mesh = "CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
  func.func private @callee(%arg0: tensor<2xi64>, %arg1: tensor<i32>) -> (tensor<2xi64>, tensor<i32>) attributes {tf.signature.is_stateful} {
    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:      %[[B_OUT:.*]] = "tf.B"
    // CHECK-NEXT:      tf_device.return %[[B_OUT]]
    // CHECK-NEXT:   _mesh = "CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
    %1 = "tf_device.cluster"() ({
      %0 = "tf.B"(%arg0, %arg1) : (tensor<2xi64>, tensor<i32>) -> (tensor<2xi64>)
      tf_device.return %0 : tensor<2xi64>
    }) : () -> tensor<2xi64>
    func.return %1, %arg1: tensor<2xi64>, tensor<i32>
  }
}

// -----

// Check mesh is propagate from function body if no mesh can be find from inputs.
// CHECK-LABEL: module @test_no_mesh_from_inputs
module @test_no_mesh_from_inputs {
  func.func @main(%arg0: tensor<i32>) -> tensor<f32> {
    %0 = "tf_device.cluster"() ({
      %1 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @mesh_from_func_body} : (tensor<i32>) -> tensor<f32>
      tf_device.return %1 : tensor<f32>
      // CHECK: _mesh = "TPU|x=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"
    }) : () -> tensor<f32>
    func.return %0 : tensor<f32>
  }

  func.func private @mesh_from_func_body(%arg0: tensor<i32>) -> tensor<f32> attributes {tf.signature.is_stateful} {
    %0 = "tf_device.cluster"() ({
      %3 = "tf.Const"() {_layout = ["sharding_specs: mesh:TPU|x=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"], value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
      tf_device.return %3 : tensor<0xi32>
    }) {_mesh = "TPU|x=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"} : () -> tensor<0xi32>
    %1 = "tf_device.cluster"() ({
      %3 = "tf.RandomUniform"(%0) {_layout = ["sharding_specs: mesh:TPU|x=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"], seed = 0 : i64, seed2 = 0 : i64} : (tensor<0xi32>) -> tensor<f32>
      tf_device.return %3 : tensor<f32>
    }) {_mesh = "TPU|x=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"} : () -> tensor<f32>
    %2 = "tf_device.cluster"() ({
      %3 = "tf.Identity"(%1) {} : (tensor<f32>) -> tensor<f32>
      tf_device.return %3 : tensor<f32>
      // CHECK: _mesh = "TPU|x=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"
    }) : () -> tensor<f32>
    func.return %2 : tensor<f32>
  }
}

// -----

// CHECK-LABEL: module @test_return_const
module @test_return_const {
  func.func @main(%arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>}, %arg1: tensor<8x128x128xf32> {tf._global_shape = #tf_type.shape<8x128x128>, tf._layout = "sharding_specs:x,unsharded,unsharded, mesh:CPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1", tf._mesh = "CPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"}) -> (tensor<3xi32> {tf._global_shape = #tf_type.shape<3>}) attributes {tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0", outputs = "op_output_0"}} {
    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:   "tf.Const"
    // CHECK-NEXT:    tf_device.return
    // CHECK-NEXT:  _mesh = "CPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"
    %0 = "tf_device.cluster"() ({
      %1 = "tf.Const"() {_global_shape = [#tf_type.shape<3>], value = dense<[8, 128, 128]> : tensor<3xi32>} : () -> tensor<3xi32>
      tf_device.return {_global_shape = []} %1 : tensor<3xi32>
    }) {_global_shape = [#tf_type.shape<3>]} : () -> tensor<3xi32>
    func.return %0 : tensor<3xi32>
  }
}

// -----

// CHECK-LABEL: module @test_multi_mesh
module @test_multi_mesh {
  func.func @main(%arg0: tensor<4xi32> {tf._layout = "sharding_specs:not_sharded mesh:CPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"},
             %arg1: tensor<4xi32> {tf._layout = "sharding_specs:not_sharded mesh:CPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"}) -> (tensor<4xi32>) {
    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:    "tf.Const"
    // CHECK-NEXT:    tf_device.return
    // CHECK-NEXT:    _mesh = "CPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"
    %0 = "tf_device.cluster"() ({
      %1 = "tf.Const"() {value = dense<[8, 8, 128, 128]> : tensor<4xi32>} : () -> tensor<4xi32>
      tf_device.return %1 : tensor<4xi32>
    }) : () -> tensor<4xi32>

    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:    "tf.Add"
    // CHECK-NEXT:    tf_device.return
    // CHECK-NEXT:    _mesh = "CPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"
    %2 = "tf_device.cluster"() ({
      %3 = "tf.Add"(%arg0, %0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
      tf_device.return %3 : tensor<4xi32>
    }) : () -> tensor<4xi32>

    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:    "tf.Identity"
    // CHECK-NEXT:    tf_device.return
    // CHECK-NEXT:    _mesh = "CPU|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"
    %4 = "tf_device.cluster"() ({
      %5 = "tf.Identity"(%arg1) : (tensor<4xi32>) -> tensor<4xi32>
      tf_device.return %5 : tensor<4xi32>
    }) : () -> tensor<4xi32>

    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:    "tf.CopyToMesh"
    // CHECK-NEXT:    tf_device.return
    // CHECK-NEXT:    _mesh = "TPU|x=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"
    %6 = "tf_device.cluster"() ({
      %7 = "tf.CopyToMesh"(%2) { layout = "sharding_specs:not_sharded mesh:TPU|x=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"} : (tensor<4xi32>) -> tensor<4xi32>
      tf_device.return %7 : tensor<4xi32>
    }) { _mesh = "TPU|x=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1" } : () -> tensor<4xi32>

    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:    "tf.CopyToMesh"
    // CHECK-NEXT:    tf_device.return
    // CHECK-NEXT:    _mesh = "TPU|x=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"
    %8 = "tf_device.cluster"() ({
      %9 = "tf.CopyToMesh"(%4) { layout = "sharding_specs:not_sharded mesh:TPU|x=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"} : (tensor<4xi32>) -> tensor<4xi32>
      tf_device.return %9 : tensor<4xi32>
    }) { _mesh = "TPU|x=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1" } : () -> tensor<4xi32>

    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:    "tf.Add"
    // CHECK-NEXT:    tf_device.return
    // CHECK-NEXT:    _mesh = "TPU|x=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"
    %10 = "tf_device.cluster"() ({
      %11 = "tf.Add"(%6, %8) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
      tf_device.return %11 : tensor<4xi32>
    }) : () -> tensor<4xi32>

    func.return %10 :tensor<4xi32>
  }
}

// -----

// Checks CopyToMeshGrad is written to CopyToMesh.
// CHECK-LABEL: module @test_copy_to_mesh_grad
module @test_copy_to_mesh_grad {
  func.func @main(%arg0: tensor<4xi32> {tf._layout = "sharding_specs:not_sharded mesh:CPU|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"},
             %arg1: tensor<4xi32> {tf._layout = "sharding_specs:not_sharded mesh:TPU|x=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"}) -> (tensor<4xi32>) {

    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:    "tf.CopyToMesh"
    // CHECK-NEXT:    tf_device.return
    // CHECK-NEXT:    _mesh = "TPU|x=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"
    %0 = "tf_device.cluster"() ({
      %1 = "tf.CopyToMeshGrad"(%arg0, %arg1) { reference_layout = "sharding_specs:not_sharded mesh:TPU|x=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
      tf_device.return %1 : tensor<4xi32>
    }) : () -> tensor<4xi32>

    func.return %0 :tensor<4xi32>
  }
}

// -----

// Check mesh propagation of ops inside tf.WhileRegion op.
// CHECK-LABEL: module @test_while
module @test_while {
  func.func @main(%arg0: tensor<i32>,
    %arg1: tensor<4xf32> {tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3", tf._mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"})
  -> (tensor<4xf32> {tf._default_layout = "sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"}) attributes {tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0", outputs = "op_output_0"}} {
    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:   "tf.WhileRegion"
    // CHECK:          "tf_device.cluster"
    // CHECK-NEXT:       constant
    // CHECK-NEXT:       tf_device.return
    // CHECK-NEXT:       _mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"
    // CHECK:          "tf_device.cluster"
    // CHECK-NEXT:       "tf.NotEqual"
    // CHECK-NEXT:       tf_device.return
    // CHECK-NEXT:       _mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"
    // CHECK-NEXT:     "tf.Yield"
    // CHECK:          "tf_device.cluster"
    // CHECK-NEXT:       constant
    // CHECK-NEXT:       tf_device.return
    // CHECK-NEXT:       _mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"
    // CHECK:          "tf_device.cluster"
    // CHECK-NEXT:       "tf.Sub"
    // CHECK-NEXT:       tf_device.return
    // CHECK-NEXT:       _mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"
    // CHECK-NEXT:     "tf.Yield"
    // CHECK:      _mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"
    %15:2 = "tf_device.cluster"() ({
      %2:2 = "tf.WhileRegion"(%arg1, %arg0) ({
        ^bb0(%carg0: tensor<4xf32>, %carg1: tensor<i32>):
           %11 = "tf_device.cluster"() ({
             %limit = arith.constant dense<5> : tensor<i32>
             tf_device.return %limit : tensor<i32>
           }) : () -> tensor<i32>


           %12 = "tf_device.cluster"() ({
             %cond = "tf.NotEqual"(%carg1, %11) : (tensor<i32>, tensor<i32>) -> tensor<i1>
             tf_device.return %cond : tensor<i1>
           }) : () -> tensor<i1>

           "tf.Yield"(%12) : (tensor<i1>) -> ()
      },  {
        ^bb0(%barg0: tensor<4xf32>, %barg1: tensor<i32>):
          %13 = "tf_device.cluster"() ({
            %one = arith.constant dense<1.0> : tensor<4xf32>
            tf_device.return %one: tensor<4xf32>
           }) : () -> tensor<4xf32>

          %14 = "tf_device.cluster"() ({
            %sub = "tf.Sub"(%barg0, %13) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
            tf_device.return %sub: tensor<4xf32>
           }) : () -> tensor<4xf32>

          "tf.Yield"(%14, %barg1) : (tensor<4xf32>, tensor<i32>) -> ()
      }) {is_stateless = true} : (tensor<4xf32>, tensor<i32>) -> (tensor<4xf32>, tensor<i32>)

      tf_device.return %2#0, %2#1 : tensor<4xf32>, tensor<i32>
    }) : () -> (tensor<4xf32>, tensor<i32>)

    %16 = "tf_device.cluster"() ({
      %5 = "tf.Identity"(%15#0) : (tensor<4xf32>) -> (tensor<4xf32>)
      tf_device.return %5 : tensor<4xf32>
    }) : () -> tensor<4xf32>

    func.return %16 : tensor<4xf32>
  }
}

// -----

// Check mesh propagation of ops inside tf.IfRegion op.
// CHECK-LABEL: module @test_if
module @test_if {
  func.func @main(%arg0: tensor<i32>,
    %arg1: tensor<4xf32> {tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1",
                          tf._mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"}) -> tensor<4xf32> {

   // CHECK:       "tf_device.cluster"
   // CHECK-NEXT:    "tf.Const"
   // CHECK-NEXT:    tf_device.return
   // CHECK-NEXT:  _mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"
   // CHECK-NEXT:  "tf_device.cluster"
   // CHECK-NEXT:    "tf.Const"
   // CHECK-NEXT:    tf_device.return
   // CHECK-NEXT:  _mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"
   // CHECK-NEXT:  "tf_device.cluster"
   // CHECK-NEXT:    "tf.Const"
   // CHECK-NEXT:    tf_device.return
   // CHECK-NEXT:  _mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"
   // CHECK-NEXT:  "tf_device.cluster"
   // CHECK-NEXT:    "tf.Const"
   // CHECK-NEXT:    tf_device.return
   // CHECK-NEXT:  _mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"
   // CHECK-NEXT:  "tf_device.cluster"
   // CHECK-NEXT:    "tf.Sum"
   // CHECK-NEXT:    tf_device.return
   // CHECK-NEXT:  _mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"
   // CHECK-NEXT:  "tf_device.cluster"
   // CHECK-NEXT:    "tf.Equal"
   // CHECK-NEXT:    tf_device.return
   // CHECK-NEXT:  _mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"
   // CHECK-NEXT:  "tf_device.cluster"
   // CHECK-NEXT:    "tf.IfRegion"
   // CHECK-NEXT:      "tf_device.cluster"
   // CHECK-NEXT:        "tf.Identity"
   // CHECK-NEXT:        tf_device.return
   // CHECK-NEXT:      _mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"
   // CHECK-NEXT:      "tf_device.cluster"
   // CHECK-NEXT:        "tf.Sqrt"
   // CHECK-NEXT:        tf_device.return
   // CHECK-NEXT:      _mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"
   // CHECK-NEXT:      "tf_device.cluster"
   // CHECK-NEXT:        "tf.Relayout"
   // CHECK-NEXT:        tf_device.return
   // CHECK-NEXT:      _mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"
   // CHECK-NEXT:      "tf_device.cluster"
   // CHECK-NEXT:        "tf.DTensorLayout"
   // CHECK-NEXT:        tf_device.return
   // CHECK-NEXT:      _mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"
   // CHECK-NEXT:      "tf_device.cluster"
   // CHECK-NEXT:        "tf.Identity"
   // CHECK-NEXT:        tf_device.return
   // CHECK-NEXT:      _mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"
   // CHECK-NEXT:      "tf.Yield"
   // CHECK:           "tf_device.cluster"
   // CHECK-NEXT:        "tf.Identity"
   // CHECK-NEXT:        tf_device.return
   // CHECK-NEXT:      _mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"
   // CHECK-NEXT:      "tf_device.cluster"
   // CHECK-NEXT:        "tf.Identity"
   // CHECK-NEXT:        tf_device.return
   // CHECK-NEXT:      _mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"
   // CHECK-NEXT:      "tf.Yield"
   // CHECK-NEXT:    (tensor<i1>) -> (tensor<i1>, tensor<4xf32>)
   // CHECK-NEXT:    tf_device.return
   // CHECK-NEXT:    _mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"
   %0 = "tf_device.cluster"() ({
      %10 = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
      tf_device.return %10 : tensor<f32>
    }) : () -> tensor<f32>
    %1 = "tf_device.cluster"() ({
      %10 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
      tf_device.return %10 : tensor<1xi32>
    }) : () -> tensor<1xi32>
    %2 = "tf_device.cluster"() ({
      %10 = "tf.Const"() {value = dense<0.000000e+00> : tensor<4xf32>} : () -> tensor<4xf32>
      tf_device.return %10 : tensor<4xf32>
    }) : () -> tensor<4xf32>
    %3 = "tf_device.cluster"() ({
      %10 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
      tf_device.return %10 : tensor<i1>
    }) : () -> tensor<i1>
    %5 = "tf_device.cluster"() ({
      %10 = "tf.Sum"(%arg1, %1) {device = "", keep_dims = false} : (tensor<4xf32>, tensor<1xi32>) -> tensor<f32>
      tf_device.return %10 : tensor<f32>
    }) : () -> tensor<f32>
    %6 = "tf_device.cluster"() ({
      %10 = "tf.Equal"(%5, %0) {device = "", incompatible_shape_error = true} : (tensor<f32>, tensor<f32>) -> tensor<i1>
      tf_device.return %10 : tensor<i1>
    }) : () -> tensor<i1>
    %7:2 = "tf_device.cluster"() ({
      %10:2 = "tf.IfRegion"(%6) ({
        %11 = "tf_device.cluster"() ({
          %16 = "tf.Identity"(%3) {device = ""} : (tensor<i1>) -> tensor<i1>
          tf_device.return %16 : tensor<i1>
        }) : () -> tensor<i1>
        %12 = "tf_device.cluster"() ({
          %16 = "tf.Sqrt"(%arg1) {device = ""} : (tensor<4xf32>) -> tensor<4xf32>
          tf_device.return %16 : tensor<4xf32>
        }) : () -> tensor<4xf32>
        %13 = "tf_device.cluster"() ({
          %16 = "tf.Relayout"(%12) {device = "", layout = "sharding_specs:x, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"} : (tensor<4xf32>) -> tensor<4xf32>
          tf_device.return %16 : tensor<4xf32>
        }) : () -> tensor<4xf32>
        %14 = "tf_device.cluster"() ({
          %16 = "tf.DTensorLayout"(%13) {global_shape = #tf_type.shape<4>,  layout = #dtensor.layout<sharding_specs:x, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<4xf32>) -> tensor<4xf32>
          tf_device.return %16 : tensor<4xf32>
        }) {_mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"} : () -> tensor<4xf32>
        %15 = "tf_device.cluster"() ({
          %16 = "tf.Identity"(%14) {device = ""} : (tensor<4xf32>) -> tensor<4xf32>
          tf_device.return %16 : tensor<4xf32>
        }) : () -> tensor<4xf32>
        "tf.Yield"(%11, %15) : (tensor<i1>, tensor<4xf32>) -> ()
      },  {
        %11 = "tf_device.cluster"() ({
          %13 = "tf.Identity"(%3) {device = ""} : (tensor<i1>) -> tensor<i1>
          tf_device.return %13 : tensor<i1>
        }) : () -> tensor<i1>
        %12 = "tf_device.cluster"() ({
          %13 = "tf.Identity"(%2) {device = ""} : (tensor<4xf32>) -> tensor<4xf32>
          tf_device.return %13 : tensor<4xf32>
        }) : () -> tensor<4xf32>
        "tf.Yield"(%11, %12) : (tensor<i1>, tensor<4xf32>) -> ()
      }) {_else_func_name = "cond_false_150", _lower_using_switch_merge = true, _read_only_resource_inputs = [], _then_func_name = "cond_true_140", device = "", is_stateless = true} : (tensor<i1>) -> (tensor<i1>, tensor<4xf32>)
      tf_device.return %10#0, %10#1 : tensor<i1>, tensor<4xf32>
    }) : () -> (tensor<i1>, tensor<4xf32>)
    %8 = "tf_device.cluster"() ({
      %10 = "tf.Identity"(%7#1) {device = ""} : (tensor<4xf32>) -> tensor<4xf32>
      tf_device.return %10 : tensor<4xf32>
    }) : () -> tensor<4xf32>
    %9 = "tf_device.cluster"() ({
      %10 = "tf.Identity"(%8) {device = ""} : (tensor<4xf32>) -> tensor<4xf32>
      tf_device.return %10 : tensor<4xf32>
    }) : () -> tensor<4xf32>

    func.return %9 : tensor<4xf32>
  }
}

// -----

// Check mesh propagation of tf.WhileRegion inside tf.IfRegion op.
// This test only checks that the code doesn't crash under asan.
// Correctness check are covered by other tests.
// CHECK-LABEL: module @test_nested_while_inside_if
module @test_nested_while_inside_if {
  func.func @main(%arg0: tensor<i32>,
    %arg1: tensor<4xf32> {tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1",
                          tf._mesh = "|x=2,y=1|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"}) -> tensor<4xf32> {

   %0 = "tf_device.cluster"() ({
      %1 = "tf.Const"() {value = dense<0> : tensor<i1>} : () -> tensor<i1>
      tf_device.return %1 : tensor<i1>
    }) : () -> tensor<i1>

   %7:1 = "tf_device.cluster"() ({
      %10:1 = "tf.IfRegion"(%0) ({
          %3:2 = "tf.WhileRegion"(%arg1, %arg0) ({
            ^bb0(%carg0: tensor<4xf32>, %carg1: tensor<i32>):
               %11 = "tf_device.cluster"() ({
                 %limit = arith.constant dense<5> : tensor<i32>
                 tf_device.return %limit : tensor<i32>
               }) : () -> tensor<i32>

               %12 = "tf_device.cluster"() ({
                 %cond = "tf.NotEqual"(%carg1, %11) : (tensor<i32>, tensor<i32>) -> tensor<i1>
                 tf_device.return %cond : tensor<i1>
               }) : () -> tensor<i1>

           "tf.Yield"(%12) : (tensor<i1>) -> ()
          },  {
            ^bb0(%barg0: tensor<4xf32>, %barg1: tensor<i32>):
              %13 = "tf_device.cluster"() ({
                %one = arith.constant dense<1.0> : tensor<4xf32>
                tf_device.return %one: tensor<4xf32>
               }) : () -> tensor<4xf32>

              %14 = "tf_device.cluster"() ({
                %sub = "tf.Sub"(%barg0, %13) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
                tf_device.return %sub: tensor<4xf32>
               }) : () -> tensor<4xf32>

              "tf.Yield"(%14, %barg1) : (tensor<4xf32>, tensor<i32>) -> ()
          }) {is_stateless = true} : (tensor<4xf32>, tensor<i32>) -> (tensor<4xf32>, tensor<i32>)
        "tf.Yield"(%3#0) : (tensor<4xf32>) -> ()
      },  {
        "tf.Yield"(%arg1) : (tensor<4xf32>) -> ()
      }) {_else_func_name = "cond_false_150", _lower_using_switch_merge = true, _read_only_resource_inputs = [], _then_func_name = "cond_true_140", device = "", is_stateless = true} : (tensor<i1>) -> (tensor<4xf32>)
      tf_device.return %10#0 : tensor<4xf32>
    }) : () -> (tensor<4xf32>)

    func.return %7#0 : tensor<4xf32>
  }
}
