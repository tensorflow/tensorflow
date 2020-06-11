// RUN: tf-opt -tf-saved-model-optimize-global-tensors -split-input-file %s | FileCheck %s
//===----------------------------------------------------------------------===//
// Immutability.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module attributes {tf_saved_model.semantics}
module attributes {tf_saved_model.semantics} {

  // Test case: This test exercises marking a global tensor as immutable after it propagates
  // via set of chained calls -> f -> f_callee -> f_callee_callee

  // CHECK: "tf_saved_model.global_tensor"() {
  // CHECK-NOT: is_mutable
  // CHECK-SAME: } : () -> ()
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<42.> : tensor<f32> } : () -> ()

  func @f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v}) -> (tensor<f32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %val = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f_callee} : (tensor<!tf.resource<tensor<f32>>>) -> (tensor<f32>)
    return %val : tensor<f32>
  }

  func @f_callee(%arg0: tensor<*x!tf.resource>) -> tensor<f32> {
    %val = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f_callee_callee} : (tensor<*x!tf.resource>) -> (tensor<f32>)
    return %val : tensor<f32>
  }

  func @f_callee_callee(%arg0: tensor<*x!tf.resource>) -> tensor<f32> {
    %val = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf.resource>) -> tensor<f32>
    return %val : tensor<f32>
  }
}

// -----

// CHECK-LABEL: module attributes {tf_saved_model.semantics}
module attributes {tf_saved_model.semantics} {

  // Test case:
  // This test exercises trying to mark immutable when same func is called by multiple callers
  // with different global tensors.
  // CHECK: "tf_saved_model.global_tensor"() {
  // CHECK-NOT: is_mutable
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<42.> : tensor<f32> } : () -> ()

  // CHECK: "tf_saved_model.global_tensor"() {
  // CHECK-NOT: is_mutable
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v2", type = tensor<f32>, value = dense<42.> : tensor<f32> } : () -> ()

  func @f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v}) -> (tensor<f32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %val = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f_common} : (tensor<!tf.resource<tensor<f32>>>) -> (tensor<f32>)
    return %val : tensor<f32>
  }

  func @f2(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v2}) -> (tensor<f32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["f2"]} {
    %val = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f_common} : (tensor<!tf.resource<tensor<f32>>>) -> (tensor<f32>)
    return %val : tensor<f32>
  }

  func @f_common(%arg0: tensor<*x!tf.resource>) -> tensor<f32> {
    %val = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf.resource>) -> tensor<f32>
    return %val : tensor<f32>
  }
}

// -----

// CHECK-LABEL: module attributes {tf_saved_model.semantics}
module attributes {tf_saved_model.semantics} {

  // Test case: This test exercises immutability without explicit use
  // via ReadVariableOp

  // CHECK: "tf_saved_model.global_tensor"() {
  // CHECK-NOT: is_mutable
  // CHECK-SAME: } : () -> ()
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<42.> : tensor<f32> } : () -> ()

  func @f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v}) -> (tensor<f32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %val = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f_callee} : (tensor<!tf.resource<tensor<f32>>>) -> (tensor<f32>)
    %val_2 = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f_callee} : (tensor<!tf.resource<tensor<f32>>>) -> (tensor<f32>)
    return %val_2 : tensor<f32>
  }

  func @f_callee(%arg0: tensor<*x!tf.resource>) -> tensor<f32> {
    %cst_1 = constant dense<2.0> : tensor<f32>
    return %cst_1 : tensor<f32>
  }
}

// -----

//===----------------------------------------------------------------------===//
// Test case: Test mutation detection propagates across function calls
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module attributes {tf_saved_model.semantics}
module attributes {tf_saved_model.semantics} {
  // CHECK: "tf_saved_model.global_tensor"() {
  // CHECK-SAME: is_mutable
  // CHECK-SAME: } : () -> ()
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<42.> : tensor<f32> } : () -> ()

  // CHECK: func @f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  func @f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v}) -> (tensor<f32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %val = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f_callee} : (tensor<!tf.resource<tensor<f32>>>) -> (tensor<f32>)
    return %val : tensor<f32>
  }

  // CHECK: func @f_callee(%arg0: tensor<*x!tf.resource>) -> tensor<f32>
  func @f_callee(%arg0: tensor<*x!tf.resource>) -> tensor<f32> {
    %val = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f_callee_callee} : (tensor<*x!tf.resource>) -> (tensor<f32>)
    return %val : tensor<f32>
  }

  // CHECK: func @f_callee_callee(%arg0: tensor<*x!tf.resource>) -> tensor<f32>
  func @f_callee_callee(%arg0: tensor<*x!tf.resource>) -> tensor<f32> {
    %c0 = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
    "tf.AssignVariableOp"(%arg0, %c0) : (tensor<*x!tf.resource>, tensor<f32>) -> ()
    return %c0 : tensor<f32>
  }
}

// -----

// CHECK-LABEL: module attributes {tf_saved_model.semantics}
module attributes {tf_saved_model.semantics} {

  // Test case: The inter-procedural analysis with different types of
  // TF call ops

  // CHECK: "tf_saved_model.global_tensor"() {
  // CHECK-SAME: is_mutable
  // CHECK-SAME: } : () -> ()
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<42.> : tensor<f32> } : () -> ()

  // CHECK: func @f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  func @f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v}) -> (tensor<f32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %val = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f_callee} : (tensor<!tf.resource<tensor<f32>>>) -> (tensor<f32>)
    return %val : tensor<f32>
  }

  // CHECK: func @f_callee(%arg0: tensor<*x!tf.resource>) -> tensor<f32>
  func @f_callee(%arg0: tensor<*x!tf.resource>) -> tensor<f32> {
    %val = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f_callee_callee} : (tensor<*x!tf.resource>) -> (tensor<f32>)
    return %val : tensor<f32>
  }

  // CHECK: func @f_callee_callee(%arg0: tensor<*x!tf.resource>) -> tensor<f32>
  func @f_callee_callee(%arg0: tensor<*x!tf.resource>) -> tensor<f32> {
    %c0 = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
    "tf.AssignVariableOp"(%arg0, %c0) : (tensor<*x!tf.resource>, tensor<f32>) -> ()
    return %c0 : tensor<f32>
  }
}

// -----

// CHECK-LABEL: module attributes {tf_saved_model.semantics}
module attributes {tf_saved_model.semantics} {

  // Test case: The inter-procedural analysis does not recurse infinitely

  // CHECK: "tf_saved_model.global_tensor"() {
  // CHECK-NOT: is_mutable
  // CHECK-SAME: } : () -> ()
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<42.> : tensor<f32> } : () -> ()

  func @exported_f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v}) -> (tensor<f32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["exported_f"]} {
    %val = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f} : (tensor<!tf.resource<tensor<f32>>>) -> (tensor<f32>)
    return %val : tensor<f32>
  }


  // CHECK: func @f(%arg0: tensor<*x!tf.resource>) -> tensor<f32>
  func @f(%arg0: tensor<*x!tf.resource>) -> tensor<f32> {
    %val = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @g} : (tensor<*x!tf.resource>) -> (tensor<f32>)
    return %val : tensor<f32>
  }

  // CHECK: func @g(%arg0: tensor<*x!tf.resource>) -> tensor<f32>
  func @g(%arg0: tensor<*x!tf.resource>) -> tensor<f32> {
    %val = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f} : (tensor<*x!tf.resource>) -> (tensor<f32>)
    return %val : tensor<f32>
  }
}

// -----

// CHECK-LABEL: module attributes {tf_saved_model.semantics}
module attributes {tf_saved_model.semantics} {

  // Test case: Inter-procedural analysis with resource usage in an
  // unknown op, we assume mutating behavior and propagate that.

  // CHECK: "tf_saved_model.global_tensor"() {
  // CHECK-SAME: is_mutable
  // CHECK-SAME: } : () -> ()
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<42.> : tensor<f32> } : () -> ()

  func @exported_f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v}) -> (tensor<f32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["exported_f"]} {
    %val = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f} : (tensor<!tf.resource<tensor<f32>>>) -> (tensor<f32>)
    return %val : tensor<f32>
  }


  // CHECK: func @f(%arg0: tensor<*x!tf.resource>) -> tensor<f32>
  func @f(%arg0: tensor<*x!tf.resource>) -> tensor<f32> {
    %c0 = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
    "tf.AssignAddVariableOp"(%arg0, %c0) : (tensor<*x!tf.resource>, tensor<f32>) -> ()
    return %c0 : tensor<f32>
  }
}
