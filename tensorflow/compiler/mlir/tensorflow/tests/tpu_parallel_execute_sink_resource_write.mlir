// RUN: tf-opt %s -tf-tpu-parallel-execute-sink-resource-write | FILECHECK_OPTS="" FileCheck %s

// CHECK-LABEL: func @multiple_uses
// CHECK-SAME:  ({{.+}}: tensor<i1>, [[ARG1:%.+]]: tensor<!tf.resource>)
func @multiple_uses(%arg0: tensor<i1>, %arg1: tensor<!tf.resource>) -> tensor<i1> {
  // CHECK:      [[PARALLEL_EXECUTE:%.+]]:2 = "tf_device.parallel_execute"
  %0:2 = "tf_device.parallel_execute"() ( {
    tf_device.return %arg0 : tensor<i1>
  }, {
    tf_device.return %arg0 : tensor<i1>
  // CHECK:      }) : () -> (tensor<i1>, tensor<i1>)
  }) : () -> (tensor<i1>, tensor<i1>)
  // CHECK-NEXT: "tf.AssignVariableOp"([[ARG1]], [[PARALLEL_EXECUTE]]#0)
  "tf.AssignVariableOp"(%arg1, %0#0) : (tensor<!tf.resource>, tensor<i1>) -> ()
  // CHECK-NEXT: return [[PARALLEL_EXECUTE]]#0
  return %0#0 : tensor<i1>
}

// CHECK-LABEL: func @not_assign_var
// CHECK-SAME:  ({{.+}}: tensor<i1>, [[ARG1:%.+]]: tensor<!tf.resource>)
func @not_assign_var(%arg0: tensor<i1>, %arg1: tensor<!tf.resource>) {
  // CHECK:      [[PARALLEL_EXECUTE:%.+]]:2 = "tf_device.parallel_execute"
  %0:2 = "tf_device.parallel_execute"() ( {
    tf_device.return %arg0 : tensor<i1>
  }, {
    tf_device.return %arg0 : tensor<i1>
  // CHECK:      }) : () -> (tensor<i1>, tensor<i1>)
  }) : () -> (tensor<i1>, tensor<i1>)
  // CHECK-NEXT: "tf.AssignAddVariableOp"([[ARG1]], [[PARALLEL_EXECUTE]]#0)
  "tf.AssignAddVariableOp"(%arg1, %0#0) : (tensor<!tf.resource>, tensor<i1>) -> ()
  return
}

// CHECK-LABEL: func @resource_handle_output
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i1>, {{.+}}: tensor<!tf.resource>)
func @resource_handle_output(%arg0: tensor<i1>, %arg1: tensor<!tf.resource>) {
  // CHECK:      [[PARALLEL_EXECUTE:%.+]]:2 = "tf_device.parallel_execute"
  %0:2 = "tf_device.parallel_execute"() ( {
    tf_device.return %arg1 : tensor<!tf.resource>
  }, {
    tf_device.return %arg1 : tensor<!tf.resource>
  // CHECK:      }) : () -> (tensor<!tf.resource>, tensor<!tf.resource>)
  }) : () -> (tensor<!tf.resource>, tensor<!tf.resource>)
  // CHECK-NEXT: "tf.AssignVariableOp"([[PARALLEL_EXECUTE]]#0, [[ARG0]])
  "tf.AssignVariableOp"(%0#0, %arg0) : (tensor<!tf.resource>, tensor<i1>) -> ()
  return
}

// CHECK-LABEL: func @resource_handle_and_value_output
func @resource_handle_and_value_output(%arg0: tensor<i1>, %arg1: tensor<!tf.resource>) {
  // CHECK: [[PARALLEL_EXECUTE:%.+]]:2 = "tf_device.parallel_execute"
  %0:2 = "tf_device.parallel_execute"() ( {
    tf_device.return %arg0, %arg1 : tensor<i1>, tensor<!tf.resource>
  }, {
    tf_device.return
  }) : () -> (tensor<i1>, tensor<!tf.resource>)
  // CHECK: "tf.AssignVariableOp"([[PARALLEL_EXECUTE]]#1, [[PARALLEL_EXECUTE]]#0)
  "tf.AssignVariableOp"(%0#1, %0#0) : (tensor<!tf.resource>, tensor<i1>) -> ()
  return
}

// CHECK-LABEL: func @resource_handle_after_parallel_execute
func @resource_handle_after_parallel_execute(%arg0: tensor<i1>) {
  // CHECK:      [[PARALLEL_EXECUTE:%.+]]:2 = "tf_device.parallel_execute"
  %0:2 = "tf_device.parallel_execute"() ( {
    tf_device.return %arg0 : tensor<i1>
  }, {
    tf_device.return %arg0 : tensor<i1>
  // CHECK:      }) : () -> (tensor<i1>, tensor<i1>)
  }) : () -> (tensor<i1>, tensor<i1>)
  // CHECK-NEXT: [[VAR:%.+]] = "tf.VarHandleOp"
  %1 = "tf.VarHandleOp"() {container = "", shape = #tf.shape<>, shared_name = "x"} : () -> tensor<!tf.resource<tensor<i1>>>
  // CHECK-NEXT: "tf.AssignVariableOp"([[VAR]], [[PARALLEL_EXECUTE]]#0)
  "tf.AssignVariableOp"(%1, %0#0) : (tensor<!tf.resource<tensor<i1>>>, tensor<i1>) -> ()
  return
}

// CHECK-LABEL: func @replace_single_output
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i1>, [[ARG1:%.+]]: tensor<i1>, [[ARG2:%.+]]: tensor<i1>, [[ARG3:%.+]]: tensor<!tf.resource>)
func @replace_single_output(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<i1>, %arg3: tensor<!tf.resource>) {
  // CHECK:      {{%.+}}:2 = "tf_device.parallel_execute"
  %0:3 = "tf_device.parallel_execute"() ( {
    // CHECK-NEXT: "tf.AssignVariableOp"([[ARG3]], [[ARG1]])
    // CHECK-NEXT: tf_device.return [[ARG0]], [[ARG2]] : tensor<i1>, tensor<i1>
    tf_device.return %arg0, %arg1, %arg2 : tensor<i1>, tensor<i1>, tensor<i1>
  // CHECK-NEXT: }, {
  }, {
    // CHECK-NEXT: tf_device.return
    tf_device.return
  // CHECK-NEXT: }) : () -> (tensor<i1>, tensor<i1>)
  }) : () -> (tensor<i1>, tensor<i1>, tensor<i1>)
  "tf.AssignVariableOp"(%arg3, %0#1) : (tensor<!tf.resource>, tensor<i1>) -> ()
  // CHECK-NEXT: return
  return
}

// CHECK-LABEL: func @replace_multiple_outputs
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i1>, [[ARG1:%.+]]: tensor<i32>, [[ARG2:%.+]]: tensor<i64>, [[ARG3:%.+]]: tensor<f32>, [[ARG4:%.+]]: tensor<f64>, [[ARG5:%.+]]: tensor<!tf.resource>, [[ARG6:%.+]]: tensor<!tf.resource>)
func @replace_multiple_outputs(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i64>, %arg3: tensor<f32>, %arg4: tensor<f64>, %arg5: tensor<!tf.resource>, %arg6: tensor<!tf.resource>) {
  // CHECK:      {{%.+}}:3 = "tf_device.parallel_execute"
  %0:5 = "tf_device.parallel_execute"() ( {
    // CHECK-NEXT: "tf.AssignVariableOp"([[ARG5]], [[ARG1]])
    // CHECK-NEXT: "tf.AssignVariableOp"([[ARG6]], [[ARG3]])
    // CHECK-NEXT: tf_device.return [[ARG0]], [[ARG2]], [[ARG4]] : tensor<i1>, tensor<i64>, tensor<f64>
    tf_device.return %arg0, %arg1, %arg2, %arg3, %arg4 : tensor<i1>, tensor<i32>, tensor<i64>, tensor<f32>, tensor<f64>
  // CHECK-NEXT: }, {
  }, {
    // CHECK-NEXT: tf_device.return
    tf_device.return
  // CHECK-NEXT: }) : () -> (tensor<i1>, tensor<i64>, tensor<f64>)
  }) : () -> (tensor<i1>, tensor<i32>, tensor<i64>, tensor<f32>, tensor<f64>)
  "tf.AssignVariableOp"(%arg5, %0#1) : (tensor<!tf.resource>, tensor<i32>) -> ()
  "tf.AssignVariableOp"(%arg6, %0#3) : (tensor<!tf.resource>, tensor<f32>) -> ()
  // CHECK-NEXT: return
  return
}

// CHECK-LABEL: func @replace_multiple_outputs_regions
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i1>, [[ARG1:%.+]]: tensor<i32>, [[ARG2:%.+]]: tensor<i64>, [[ARG3:%.+]]: tensor<bf16>, [[ARG4:%.+]]: tensor<f32>, [[ARG5:%.+]]: tensor<f64>, [[ARG6:%.+]]: tensor<!tf.resource>, [[ARG7:%.+]]: tensor<!tf.resource>)
func @replace_multiple_outputs_regions(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i64>, %arg3: tensor<bf16>, %arg4: tensor<f32>, %arg5: tensor<f64>, %arg6: tensor<!tf.resource>, %arg7: tensor<!tf.resource>) {
  // CHECK:      {{%.+}}:4 = "tf_device.parallel_execute"
  %0:6 = "tf_device.parallel_execute"() ( {
    // CHECK-NEXT: "tf.AssignVariableOp"([[ARG6]], [[ARG1]])
    // CHECK-NEXT: tf_device.return [[ARG0]], [[ARG2]] : tensor<i1>, tensor<i64>
    tf_device.return %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>, tensor<i64>
  // CHECK-NEXT: }, {
  }, {
    // CHECK-NEXT: "tf.AssignVariableOp"([[ARG7]], [[ARG4]])
    // CHECK-NEXT: tf_device.return [[ARG3]], [[ARG5]] : tensor<bf16>, tensor<f64>
    tf_device.return %arg3, %arg4, %arg5 : tensor<bf16>, tensor<f32>, tensor<f64>
  // CHECK-NEXT: }) : () -> (tensor<i1>, tensor<i64>, tensor<bf16>, tensor<f64>)
  }) : () -> (tensor<i1>, tensor<i32>, tensor<i64>, tensor<bf16>, tensor<f32>, tensor<f64>)
  "tf.AssignVariableOp"(%arg6, %0#1) : (tensor<!tf.resource>, tensor<i32>) -> ()
  "tf.AssignVariableOp"(%arg7, %0#4) : (tensor<!tf.resource>, tensor<f32>) -> ()
  // CHECK-NEXT: return
  return
}
