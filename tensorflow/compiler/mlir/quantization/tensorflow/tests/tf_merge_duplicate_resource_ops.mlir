// RUN: tf-quant-opt %s -split-input-file -tf-quant-merge-duplicate-resource-ops | FileCheck %s

func.func @merge_duplicate_variable(%arg0: tensor<1x20xf32>, %arg1: tensor<!tf_type.string>) -> (tensor<20x4096xf32>) {
  %0 = tf_executor.graph {
    %outputs_5, %control_6 = tf_executor.island wraps "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %outputs_7, %control_8 = tf_executor.island wraps "tf.Const"() {value = dense<"MatMul/b_0"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %outputs_9, %control_10 = tf_executor.island wraps "tf.VarHandleOp"() {container = "", shared_name = "MatMul/b_0"} : () -> tensor<!tf_type.resource<tensor<20x4096xf32>>>
    %outputs_11, %control_12 = tf_executor.island wraps "tf.RestoreV2"(%arg1, %outputs_7, %outputs_5) : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> tensor<20x4096xf32>
    %control_13 = tf_executor.island(%control_12) wraps "tf.AssignVariableOp"(%outputs_9, %outputs_11) {validate_shape = false} : (tensor<!tf_type.resource<tensor<20x4096xf32>>>, tensor<20x4096xf32>) -> ()
    %control_14 = tf_executor.island(%control_13) wraps "tf.NoOp"() : () -> ()
    %outputs_15, %control_16 = tf_executor.island wraps "tf.VarHandleOp"() {container = "", shared_name = "MatMul/b_0"} : () -> tensor<!tf_type.resource<tensor<20x4096xf32>>>
    %outputs_17, %control_18 = tf_executor.island wraps "tf.ReadVariableOp"(%outputs_15) : (tensor<!tf_type.resource<tensor<20x4096xf32>>>) -> tensor<20x4096xf32>
    %outputs_19, %control_20 = tf_executor.island wraps "tf.Const"() {value = dense<"MatMul/b_0"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %outputs_21, %control_22 = tf_executor.island wraps "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %control_23 = tf_executor.island(%control_18) wraps "tf.SaveV2"(%arg1, %outputs_19, %outputs_21, %outputs_17) : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>, tensor<20x4096xf32>) -> ()
    %outputs_24, %control_25 = tf_executor.island(%control_23) wraps "tf.Identity"(%arg1) : (tensor<!tf_type.string>) -> tensor<!tf_type.string>
    tf_executor.fetch %outputs_17, %control_14, %control_25 : tensor<20x4096xf32>, !tf_executor.control, !tf_executor.control
  }
  return %0 : tensor<20x4096xf32>
}
// CHECK-LABEL: @merge_duplicate_variable
// CHECK: %[[OUT_0:.*]], %[[CTL_0:.*]] = tf_executor.island wraps "tf.VarHandleOp"()
// CHECK: %[[OUT_1:.*]], %[[CTL_1:.*]] = tf_executor.island wraps "tf.RestoreV2"
// CHECK: %[[CTL_2:.*]] = tf_executor.island(%[[CTL_1]]) wraps "tf.AssignVariableOp"(%[[OUT_0]], %[[OUT_1]])

// Check that ReadVariableOp now use the same variable op.
// CHECK: %[[OUT_3:.*]], %[[CTL_3:.*]] = tf_executor.island wraps "tf.ReadVariableOp"(%[[OUT_0]])

// -----

func.func @variables_with_different_shared_names(%arg0: tensor<1x20xf32>, %arg1: tensor<!tf_type.string>) -> (tensor<20x4096xf32>) {
  %0 = tf_executor.graph {
    %outputs_5, %control_6 = tf_executor.island wraps "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %outputs_7, %control_8 = tf_executor.island wraps "tf.Const"() {value = dense<"MatMul/b_0"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %outputs_9, %control_10 = tf_executor.island wraps "tf.VarHandleOp"() {container = "", shared_name = "MatMul/b_0"} : () -> tensor<!tf_type.resource<tensor<20x4096xf32>>>
    %outputs_11, %control_12 = tf_executor.island wraps "tf.RestoreV2"(%arg1, %outputs_7, %outputs_5) : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> tensor<20x4096xf32>
    %control_13 = tf_executor.island(%control_12) wraps "tf.AssignVariableOp"(%outputs_9, %outputs_11) {validate_shape = false} : (tensor<!tf_type.resource<tensor<20x4096xf32>>>, tensor<20x4096xf32>) -> ()
    %control_14 = tf_executor.island(%control_13) wraps "tf.NoOp"() : () -> ()
    %outputs_15, %control_16 = tf_executor.island wraps "tf.VarHandleOp"() {container = "", shared_name = "MatMul/b_1"} : () -> tensor<!tf_type.resource<tensor<20x4096xf32>>>
    %outputs_17, %control_18 = tf_executor.island wraps "tf.ReadVariableOp"(%outputs_15) : (tensor<!tf_type.resource<tensor<20x4096xf32>>>) -> tensor<20x4096xf32>
    %outputs_19, %control_20 = tf_executor.island wraps "tf.Const"() {value = dense<"MatMul/b_0"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %outputs_21, %control_22 = tf_executor.island wraps "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %control_23 = tf_executor.island(%control_18) wraps "tf.SaveV2"(%arg1, %outputs_19, %outputs_21, %outputs_17) : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>, tensor<20x4096xf32>) -> ()
    %outputs_24, %control_25 = tf_executor.island(%control_23) wraps "tf.Identity"(%arg1) : (tensor<!tf_type.string>) -> tensor<!tf_type.string>
    tf_executor.fetch %outputs_17, %control_14, %control_25 : tensor<20x4096xf32>, !tf_executor.control, !tf_executor.control
  }
  return %0 : tensor<20x4096xf32>
}
// CHECK-LABEL: @variables_with_different_shared_names
// CHECK: %[[OUT_0:.*]], %[[CTL_0:.*]] = tf_executor.island wraps "tf.VarHandleOp"()
// CHECK-SAME: shared_name = "MatMul/b_0"
// CHECK: %[[OUT_1:.*]], %[[CTL_1:.*]] = tf_executor.island wraps "tf.RestoreV2"
// CHECK: %[[CTL_2:.*]] = tf_executor.island(%[[CTL_1]]) wraps "tf.AssignVariableOp"(%[[OUT_0]], %[[OUT_1]])

// Check that the second variable is not removed since they have different
// `shared_name` attribute.
// CHECK: %[[OUT_3:.*]], %[[CTL_3:.*]] = tf_executor.island wraps "tf.VarHandleOp"()
// CHECK-SAME: shared_name = "MatMul/b_1"
// CHECK: %[[OUT_4:.*]], %[[CTL_4:.*]] = tf_executor.island wraps "tf.ReadVariableOp"(%[[OUT_3]])

// -----

// Test two resource ops have the same shared_name but different types.
// expected-error @+1 {{This op has the same `shared_name` but different type with another}}
func.func @same_shared_name_but_different_types(%arg0: tensor<1x20xf32>, %arg1: tensor<!tf_type.string>) -> (tensor<20x4096xf32>) {
  %0 = tf_executor.graph {
    %outputs_5, %control_6 = tf_executor.island wraps "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %outputs_7, %control_8 = tf_executor.island wraps "tf.Const"() {value = dense<"MatMul/b_0"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %outputs_9, %control_10 = tf_executor.island wraps "tf.VarHandleOp"() {container = "", shared_name = "MatMul/b_0"} : () -> tensor<!tf_type.resource<tensor<20x4096xf32>>>
    %outputs_11, %control_12 = tf_executor.island wraps "tf.RestoreV2"(%arg1, %outputs_7, %outputs_5) : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> tensor<20x4096xf32>
    %control_13 = tf_executor.island(%control_12) wraps "tf.AssignVariableOp"(%outputs_9, %outputs_11) {validate_shape = false} : (tensor<!tf_type.resource<tensor<20x4096xf32>>>, tensor<20x4096xf32>) -> ()
    %control_14 = tf_executor.island(%control_13) wraps "tf.NoOp"() : () -> ()
    %outputs_15, %control_16 = tf_executor.island wraps "tf.VarHandleOp"() {container = "", shared_name = "MatMul/b_0"} : () -> tensor<!tf_type.resource<tensor<20x4096xf32>>>
    %outputs_17, %control_18 = tf_executor.island wraps "tf.ReadVariableOp"(%outputs_15) : (tensor<!tf_type.resource<tensor<20x4096xf32>>>) -> tensor<20x4096xf32>
    %outputs_19, %control_20 = tf_executor.island wraps "tf.Const"() {value = dense<"MatMul/b_0"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %outputs_21, %control_22 = tf_executor.island wraps "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %control_23 = tf_executor.island(%control_18) wraps "tf.SaveV2"(%arg1, %outputs_19, %outputs_21, %outputs_17) : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>, tensor<20x4096xf32>) -> ()
    %outputs_24, %control_25 = tf_executor.island(%control_23) wraps "tf.Identity"(%arg1) : (tensor<!tf_type.string>) -> tensor<!tf_type.string>
    tf_executor.fetch %outputs_17, %control_14, %control_25 : tensor<20x4096xf32>, !tf_executor.control, !tf_executor.control
  }
  return %0 : tensor<20x4096xf32>
}

// -----

func.func @merge_hashtable_ops(%arg0: tensor<?x!tf_type.string>) -> (tensor<i64>) {
  %0 = tf_executor.graph {
    %outputs, %control = tf_executor.island wraps "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "hash_table_ce3dfbfc-7367-4d62-9d48-d13bf8125391", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
    %outputs_0, %control_1 = tf_executor.island wraps "tf.LookupTableSizeV2"(%outputs) {device = ""} : (tensor<!tf_type.resource>) -> tensor<i64>
    %outputs_2, %control_3 = tf_executor.island wraps "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %outputs_4, %control_5 = tf_executor.island wraps "tf.Identity"(%outputs_0) : (tensor<i64>) -> tensor<i64>
    %control_8 = tf_executor.island(%control_3, %control_5) wraps "tf.NoOp"() : () -> ()
    %outputs_9, %control_10 = tf_executor.island wraps "tf.Const"() {value = dense<["hello", "model", "quantization"]> : tensor<3x!tf_type.string>} : () -> tensor<3x!tf_type.string>
    %outputs_11, %control_12 = tf_executor.island wraps "tf.Const"() {value = dense<[0, 1, 2]> : tensor<3xi64>} : () -> tensor<3xi64>
    %outputs_13, %control_14 = tf_executor.island wraps "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "hash_table_ce3dfbfc-7367-4d62-9d48-d13bf8125391", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
    %control_15 = tf_executor.island wraps "tf.LookupTableImportV2"(%outputs_13, %outputs_9, %outputs_11) {_has_manual_control_dependencies = true, device = ""} : (tensor<!tf_type.resource>, tensor<3x!tf_type.string>, tensor<3xi64>) -> ()
    %control_16 = tf_executor.island(%control_15) wraps "tf.NoOp"() : () -> ()
    tf_executor.fetch %outputs_4, %control_8, %control_16 : tensor<i64>, !tf_executor.control, !tf_executor.control
  }
  return %0 : tensor<i64>
}

// CHECK-LABEL: @merge_hashtable_ops
// CHECK: %[[OUT_0:.*]], %[[CTL_0:.*]] = tf_executor.island wraps "tf.HashTableV2"()
// CHECK: %[[OUT_1:.*]], %[[CTL_1:.*]] = tf_executor.island wraps "tf.LookupTableSizeV2"(%[[OUT_0]])

// Check that LookupTableImportV2 is using the same HashTableV2 with LookupTableSizeV2.
// CHECK: %[[CTL_2:.*]] = tf_executor.island wraps "tf.LookupTableImportV2"(%[[OUT_0]]
