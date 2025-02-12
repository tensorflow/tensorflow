// RUN: tf-opt --split-input-file -tf-lower-to-mlprogram-and-hlo %s -o - | FileCheck %s

module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: func @lowers_to_stablehlo
  func.func @lowers_to_stablehlo(%arg0: tensor<i32> {tf_saved_model.index_path = []}) -> (tensor<i32> {tf_saved_model.index_path = []})
    attributes {tf_saved_model.exported_names = ["lowers_to_stablehlo"]}
  {
    // CHECK-DAG: [[one:%.*]] = stablehlo.constant dense<1>
    // CHECK-DAG: [[twenty:%.*]] = stablehlo.constant dense<20>
    // CHECK-DAG: [[r3:%.*]] = stablehlo.subtract [[twenty]], %arg0
    // CHECK-DAG: [[zero:%.*]] = stablehlo.constant dense<0>
    // CHECK-DAG: [[r4:%.*]] = stablehlo.divide [[r3]], [[one]]
    // CHECK-DAG: [[r5:%.*]] = stablehlo.compare NE
    // CHECK-DAG: [[r6:%.*]] = stablehlo.compare LT
    // CHECK: [[result:%.*]] = stablehlo.select
    // CHECK-NEXT: return [[result]]
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.Const"() {device = "", value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<20> : tensor<i32>} : () -> tensor<i32>
      %outputs_2, %control_3 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %outputs_4, %control_5 = tf_executor.island wraps "tf.Range"(%outputs_2, %outputs_0, %outputs) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
      %outputs_6, %control_7 = tf_executor.island wraps "tf.Sub"(%outputs_0, %arg0) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %outputs_8, %control_9 = tf_executor.island wraps "tf.FloorDiv"(%outputs_6, %outputs) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.fetch %outputs_8 : tensor<i32>
    }
    func.return %0 : tensor<i32>
  }
}

// -----

module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: func @removes_dead_code
  func.func @removes_dead_code(%arg0: tensor<?x!tf_type.resource> {tf._user_specified_name = "iterator", tf.device = "/job:localhost/replica:0/task:0/device:CPU:0", tf_saved_model.index_path = []})
    attributes {tf_saved_model.exported_names = ["removes_dead_code"]}
  {
    // CHECK-NEXT: return
    tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.Const"() {device = "", value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<20> : tensor<i32>} : () -> tensor<i32>
      %outputs_2, %control_3 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %outputs_4, %control_5 = tf_executor.island wraps "tf.Range"(%outputs_2, %outputs_0, %outputs) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
      %outputs_6, %control_7 = tf_executor.island wraps "tf.Sub"(%outputs_0, %outputs_2) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %outputs_8, %control_9 = tf_executor.island wraps "tf.FloorDiv"(%outputs_6, %outputs) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_executor.fetch %control_9 : !tf_executor.control
    }
    return
  }
}

// -----

module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: func @lowers_variable_ops
  func.func @lowers_variable_ops()
    attributes {tf_saved_model.exported_names = ["lowers_variable_ops"]}
  {
    // CHECK: ml_program.global_store
    tf_executor.graph {
      %0, %c0 = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %1, %c1 = tf_executor.island wraps "tf.VarHandleOp"() {container = "", shared_name = "v"} : () -> tensor<!tf_type.resource<tensor<i32>>>
      %c2 = tf_executor.island wraps "tf.AssignVariableOp"(%1, %0) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
      tf_executor.fetch %c2 : !tf_executor.control
    }
    return
  }
}

// -----

module attributes {tf_saved_model.semantics} {

  func.func private @while_cond(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i1> {
    %0 = tf_executor.graph {
      %1, %c0 = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<i1>} : () -> tensor<i1>
      tf_executor.fetch %1 : tensor<i1>
    }
    return %0 : tensor<i1>
  }

  func.func private @while_body(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
    tf_executor.graph {
      %1, %c1 = tf_executor.island wraps "tf.VarHandleOp"() {container = "", shared_name = "v"} : () -> tensor<!tf_type.resource<tensor<i32>>>
      %c2 = tf_executor.island wraps "tf.AssignVariableOp"(%1, %arg1) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
      tf_executor.fetch %c2 : !tf_executor.control
    }
    return %arg0, %arg1 : tensor<i32>, tensor<i32>
  }

  // CHECK-LABEL: func @handles_variables_in_while_loops
  func.func @handles_variables_in_while_loops(%arg0: tensor<!tf_type.resource<tensor<i32>>> {tf._user_specified_name = "arg0", tf.device = "/job:localhost/replica:0/task:0/device:CPU:0", tf_saved_model.index_path = []})
    attributes {tf_saved_model.exported_names = ["handles_variables_in_while_loops"]}
  {
    // CHECK: stablehlo.while
    tf_executor.graph {
      %0, %c0 = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %1, %c1 = tf_executor.island wraps "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %2, %3, %c2 = tf_executor.island wraps "tf.While"(%0, %1) {
          body = @while_body , cond = @while_cond, is_stateless = true
      } : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
      tf_executor.fetch %c2 : !tf_executor.control
    }
    return
  }
}

// -----

module attributes {tf_saved_model.semantics} {
  // CHECK: func @a
  // CHECK: func @b
  // CHECK: func @c
  // CHECK-LABEL @expands_exported_names
  func.func @expands_exported_names()
      attributes {tf_saved_model.exported_names = ["a", "b", "c"]} {
    return
  }
}

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1512 : i32}} {
  // CHECK-LABEL @lowers_string_ops
  // CHECK-DAG: ml_program.global public @vars.Variable_1([]) : tensor<!tf_type.string>
  func.func @lowers_string_ops(%arg0: tensor<128xi32>, %arg1: tensor<128xi32>, %arg2: tensor<128x1xi32>, %arg3: tensor<128x90xi32>, %arg4: tensor<128x90xi32>, %arg5: tensor<128x90xi32>, %arg6: tensor<128x90x64xf32>, %arg7: tensor<128x90x64xf32>) -> tensor<!tf_type.string> {
    // CHECK: %[[v0:.*]] = ml_program.global_load @vars.Variable_1 : tensor<!tf_type.string>
    %0 = tf_executor.graph {
      %outputs_4, %control_5 = tf_executor.island wraps "tf.VarHandleOp"() {container = "", shared_name = "Variable"} : () -> tensor<!tf_type.resource<tensor<!tf_type.string>>>
      %outputs_10, %control_11 = tf_executor.island wraps "tf.VarHandleOp"() {container = "", shared_name = "Variable_1"} : () -> tensor<!tf_type.resource<tensor<!tf_type.string>>>
      %outputs_14, %control_15 = tf_executor.island wraps "tf.ReadVariableOp"(%outputs_10) : (tensor<!tf_type.resource<tensor<!tf_type.string>>>) -> tensor<!tf_type.string>
      %control_18 = tf_executor.island wraps "tf.AssignVariableOp"(%outputs_4, %outputs_14) : (tensor<!tf_type.resource<tensor<!tf_type.string>>>, tensor<!tf_type.string>) -> ()
      tf_executor.fetch %outputs_14 : tensor<!tf_type.string>
   }
   func.return %0 : tensor<!tf_type.string>
  }
}
