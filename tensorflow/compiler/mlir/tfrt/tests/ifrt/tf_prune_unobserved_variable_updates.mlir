// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: tf-tfrt-opt --split-input-file -tf-prune-unobserved-variable-updates %s | FileCheck %s

// A Keras-metric-style accumulator: read, add, write back to the same
// variable, value never observed anywhere else. The whole cycle (and the
// then-unused handle) is pruned. The restore in the init function stays.

// CHECK-LABEL: func.func @restore_init
// CHECK:         "tf.IfrtRestoreVariableOp"
func.func @restore_init() {
  %cst = "tf.Const"() <{value = dense<"prefix"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
  %cst_0 = "tf.Const"() <{value = dense<""> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
  %cst_1 = "tf.Const"() <{value = dense<"total_4"> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
  %0 = "tf.VarHandleOp"() <{container = "", shared_name = "total_4"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
  "tf.IfrtRestoreVariableOp"(%cst, %cst_1, %cst_0, %0) <{restored_dtypes = [f32], returned_tensor_names = [], truncate_in_cast = array<i1: false>}> : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>, tensor<!tf_type.resource<tensor<f32>>>) -> ()
  return
}

// CHECK-LABEL: func.func @serving_default
// CHECK-NOT:     tf.ReadVariableOp
// CHECK-NOT:     tf.AssignVariableOp
// CHECK:         return
func.func @serving_default(%arg0: tensor<f32>) -> tensor<f32> {
  %handle = "tf.VarHandleOp"() <{container = "", shared_name = "total_4"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
  %read = "tf.ReadVariableOp"(%handle) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %sum = "tf.AddV2"(%read, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "tf.AssignVariableOp"(%handle, %sum) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  %out = "tf.AddV2"(%arg0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %out : tensor<f32>
}

// -----

// The accumulated value escapes through the function return: nothing is
// pruned.

// CHECK-LABEL: func.func @value_returned
// CHECK:         tf.ReadVariableOp
// CHECK:         tf.AddV2
// CHECK:         tf.AssignVariableOp
func.func @value_returned(%arg0: tensor<f32>) -> tensor<f32> {
  %handle = "tf.VarHandleOp"() <{container = "", shared_name = "total_4"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
  %read = "tf.ReadVariableOp"(%handle) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %sum = "tf.AddV2"(%read, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "tf.AssignVariableOp"(%handle, %sum) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  return %sum : tensor<f32>
}

// -----

// Another function reads the variable and returns it: nothing is pruned in
// either function.

// CHECK-LABEL: func.func @serving_with_metrics
// CHECK:         tf.ReadVariableOp
// CHECK:         tf.AssignVariableOp
func.func @serving_with_metrics(%arg0: tensor<f32>) -> tensor<f32> {
  %handle = "tf.VarHandleOp"() <{container = "", shared_name = "total_4"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
  %read = "tf.ReadVariableOp"(%handle) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %sum = "tf.AddV2"(%read, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "tf.AssignVariableOp"(%handle, %sum) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  %out = "tf.AddV2"(%arg0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %out : tensor<f32>
}

// CHECK-LABEL: func.func @metrics
// CHECK:         tf.ReadVariableOp
func.func @metrics() -> tensor<f32> {
  %handle = "tf.VarHandleOp"() <{container = "", shared_name = "total_4"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
  %read = "tf.ReadVariableOp"(%handle) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  return %read : tensor<f32>
}

// -----

// A raw AssignAddVariableOp (not decomposed) is an unknown handle user for
// this analysis and must fail closed: nothing is pruned.

// CHECK-LABEL: func.func @raw_assign_add
// CHECK:         tf.AssignAddVariableOp
func.func @raw_assign_add(%arg0: tensor<f32>) {
  %handle = "tf.VarHandleOp"() <{container = "", shared_name = "total_4"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
  "tf.AssignAddVariableOp"(%handle, %arg0) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  return
}

// -----

// The read result feeds a side-effecting op (IfrtCall): the value escapes,
// nothing is pruned.

// CHECK-LABEL: func.func @read_feeds_ifrt_call
// CHECK:         tf.ReadVariableOp
// CHECK:         tf.AssignVariableOp
func.func @read_feeds_ifrt_call(%arg0: tensor<f32>) -> tensor<f32> {
  %handle = "tf.VarHandleOp"() <{container = "", shared_name = "w"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
  %read = "tf.ReadVariableOp"(%handle) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %result = "tf.IfrtCall"(%read) <{operandSegmentSizes = array<i32: 1, 0>, program_id = 6515870160938153680 : i64, variable_arg_indices = []}> : (tensor<f32>) -> tensor<f32>
  "tf.AssignVariableOp"(%handle, %arg0) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  return %result : tensor<f32>
}

// -----

// A resource-typed function argument cannot be attributed to a VarHandleOp:
// pruning is disabled for the whole module, so even the well-formed
// accumulator in the sibling function is kept.

// CHECK-LABEL: func.func @has_resource_arg
func.func @has_resource_arg(%arg0: tensor<!tf_type.resource<tensor<f32>>>) {
  return
}

// CHECK-LABEL: func.func @accumulator_kept
// CHECK:         tf.ReadVariableOp
// CHECK:         tf.AssignVariableOp
func.func @accumulator_kept(%arg0: tensor<f32>) {
  %handle = "tf.VarHandleOp"() <{container = "", shared_name = "total_4"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
  %read = "tf.ReadVariableOp"(%handle) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %sum = "tf.AddV2"(%read, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "tf.AssignVariableOp"(%handle, %sum) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  return
}

// -----

// Traced-restore style: the accumulator is updated inside a called helper
// that receives the handle as a function argument. The argument is
// attributed through the call site, so the cycle inside the callee is
// pruned.

// CHECK-LABEL: func.func @caller_with_handle_arg
func.func @caller_with_handle_arg(%arg0: tensor<f32>) {
  %handle = "tf.VarHandleOp"() <{container = "", shared_name = "total_4"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
  "tf.StatefulPartitionedCall"(%handle, %arg0) <{config = "", config_proto = "", executor_type = "", f = @update_total}> : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  return
}

// CHECK-LABEL: func.func private @update_total
// CHECK-NOT:     tf.ReadVariableOp
// CHECK-NOT:     tf.AssignVariableOp
// CHECK:         return
func.func private @update_total(%arg0: tensor<!tf_type.resource<tensor<f32>>>, %arg1: tensor<f32>) {
  %read = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %sum = "tf.AddV2"(%read, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "tf.AssignVariableOp"(%arg0, %sum) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  return
}

// -----

// Two call sites pass DIFFERENT variables for the same argument: the
// argument cannot be attributed to a single variable, so the global gate
// fires and nothing is pruned anywhere.

// CHECK-LABEL: func.func @two_callers
func.func @two_callers(%arg0: tensor<f32>) {
  %a = "tf.VarHandleOp"() <{container = "", shared_name = "total_a"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
  %b = "tf.VarHandleOp"() <{container = "", shared_name = "total_b"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
  "tf.StatefulPartitionedCall"(%a, %arg0) <{config = "", config_proto = "", executor_type = "", f = @update_shared}> : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  "tf.StatefulPartitionedCall"(%b, %arg0) <{config = "", config_proto = "", executor_type = "", f = @update_shared}> : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  return
}

// CHECK-LABEL: func.func private @update_shared
// CHECK:         tf.ReadVariableOp
// CHECK:         tf.AssignVariableOp
func.func private @update_shared(%arg0: tensor<!tf_type.resource<tensor<f32>>>, %arg1: tensor<f32>) {
  %read = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %sum = "tf.AddV2"(%read, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "tf.AssignVariableOp"(%arg0, %sum) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  return
}

// -----

// Two accumulators (Keras Mean keeps total and count): both cycles are
// pruned independently.

// CHECK-LABEL: func.func @mean_metric
// CHECK-NOT:     tf.ReadVariableOp
// CHECK-NOT:     tf.AssignVariableOp
// CHECK:         return
func.func @mean_metric(%arg0: tensor<f32>) -> tensor<f32> {
  %total = "tf.VarHandleOp"() <{container = "", shared_name = "total_4"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
  %count = "tf.VarHandleOp"() <{container = "", shared_name = "count_4"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
  %one = "tf.Const"() <{value = dense<1.0> : tensor<f32>}> : () -> tensor<f32>
  %read_total = "tf.ReadVariableOp"(%total) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %new_total = "tf.AddV2"(%read_total, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "tf.AssignVariableOp"(%total, %new_total) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  %read_count = "tf.ReadVariableOp"(%count) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %new_count = "tf.AddV2"(%read_count, %one) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "tf.AssignVariableOp"(%count, %new_count) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  %out = "tf.AddV2"(%arg0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %out : tensor<f32>
}

// -----

// A non-variable resource op (e.g. tf.HashTableV2) does not trigger the
// unattributed resource check and does not prevent pruning of accumulators.

// CHECK-LABEL: func.func @has_hash_table_and_accumulator
// CHECK-NOT:     tf.ReadVariableOp
// CHECK-NOT:     tf.AssignVariableOp
// CHECK:         tf.HashTableV2
// CHECK:         return
func.func @has_hash_table_and_accumulator(%arg0: tensor<f32>) -> tensor<f32> {
  %hash_table = "tf.HashTableV2"() <{container = "", key_dtype = i64, shared_name = "hash_table", use_node_name_sharing = false, value_dtype = i64}> : () -> tensor<!tf_type.resource>
  %handle = "tf.VarHandleOp"() <{container = "", shared_name = "total_4"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
  %read = "tf.ReadVariableOp"(%handle) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %sum = "tf.AddV2"(%read, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "tf.AssignVariableOp"(%handle, %sum) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  return %arg0 : tensor<f32>
}

// -----

// A variable checked with VarIsInitializedOp or closed with DestroyResourceOp
// is not considered observed, so its accumulator cycle is still pruned.

// CHECK-LABEL: func.func @var_is_initialized
// CHECK-NOT:     tf.ReadVariableOp
// CHECK-NOT:     tf.AssignVariableOp
// CHECK:         return
func.func @var_is_initialized(%arg0: tensor<f32>) -> tensor<f32> {
  %handle = "tf.VarHandleOp"() <{container = "", shared_name = "total_4"}> : () -> tensor<!tf_type.resource<tensor<f32>>>
  %init = "tf.VarIsInitializedOp"(%handle) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<i1>
  %read = "tf.ReadVariableOp"(%handle) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
  %sum = "tf.AddV2"(%read, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "tf.AssignVariableOp"(%handle, %sum) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  "tf.DestroyResourceOp"(%handle) <{ignore_lookup_error = true}> : (tensor<!tf_type.resource<tensor<f32>>>) -> ()
  return %arg0 : tensor<f32>
}
