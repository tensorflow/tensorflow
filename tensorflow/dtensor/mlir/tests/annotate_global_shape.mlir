// RUN: dtensor-opt %s -dtensor-annotate-global-shape -split-input-file | FileCheck %s

// CHECK-LABEL: func @check_op_global_shape_annotated
func.func @check_op_global_shape_annotated() {
  // CHECK:      "tf.A"() {_global_shape = [#tf_type.shape<>]}
  // CHECK-NEXT: "tf.B"() {_global_shape = [#tf_type.shape<64x64>, #tf_type.shape<2x8x8>]}
  %1 = "tf.A"() : () -> tensor<i32>
  %2, %3 = "tf.B"() : () -> (tensor<64x64xi64>, tensor<2x8x8xf32>)
  func.return
}

// -----

// CHECK-LABEL: func @check_op_with_unranked_type_annotated
func.func @check_op_with_unranked_type_annotated() {
  // CHECK:      "tf.B"() {_global_shape = [#tf_type.shape<*>]}
  %1 = "tf.B"() : () -> tensor<*xi32>
  func.return
}

// -----

// CHECK-LABEL: func @check_op_with_non_static_shape
func.func @check_op_with_non_static_shape() {
  // CHECK: "tf.B"() {_global_shape = [#tf_type.shape<4>, #tf_type.shape<?>]}
  %1, %2 = "tf.B"() : () -> (tensor<4xi32>, tensor<?xi32>)
  func.return
}

// -----

// CHECK-LABEL: func @check_function_arg_retval_annotated
// CHECK-SAME:  %arg0: tensor<4x2xi32> {tf._global_shape = #tf_type.shape<4x2>}
// CHECK-SAME: (tensor<4x2xi32> {tf._global_shape = #tf_type.shape<4x2>})
func.func @check_function_arg_retval_annotated(%arg0: tensor<4x2xi32>) -> tensor<4x2xi32> {
  %0 = "tf.Identity"(%arg0) : (tensor<4x2xi32>) -> tensor<4x2xi32>
  func.return %0 : tensor<4x2xi32>
}

// -----

// CHECK-LABEL: func @check_function_callsites_annotated_properly
// CHECK-SAME:  %arg0: tensor<4x2xi32> {tf._global_shape = #tf_type.shape<4x2>}
// CHECK-SAME: (tensor<4x2xi32> {tf._global_shape = #tf_type.shape<4x2>})
func.func @check_function_callsites_annotated_properly(%arg0: tensor<4x2xi32>) -> tensor<4x2xi32> {
  // CHECK:      "tf.StatefulPartitionedCall"
  // CHECK-SAME: _global_shape = [#tf_type.shape<4x2>]
  // CHECK-SAME: (tensor<4x2xi32>) -> tensor<4x2xi32>
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @call_func} : (tensor<4x2xi32>) -> (tensor<4x2xi32>)
  func.return %0 : tensor<4x2xi32>
}

// CHECK-LABEL: func @call_func
// CHECK-SAME:  %arg0: tensor<4x2xi32> {tf._global_shape = #tf_type.shape<4x2>}
func.func @call_func(%arg0: tensor<4x2xi32>) -> tensor<4x2xi32> {
  func.return %arg0 : tensor<4x2xi32>
}

// -----

// CHECK-LABEL: func @check_resource_type_shape
// CHECK-SAME:  %arg1: tensor<!tf_type.resource<tensor<4x2xf32>>> {tf._global_shape = #tf_type.shape<4x2>
func.func @check_resource_type_shape(%arg0: tensor<i32>, %arg1: tensor<!tf_type.resource<tensor<4x2xf32>>> {tf._layout = "sharding_specs: mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1", tf._mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"}) attributes {tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0", outputs = ""}} {
    "tf_device.cluster"() ({
      %0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
      "tf.AssignVariableOp"(%arg1, %0) {device = ""} : (tensor<!tf_type.resource<tensor<4x2xf32>>>, tensor<f32>) -> ()
      tf_device.return
    }) {_mesh = "|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"} : () -> ()
    func.return
  }
