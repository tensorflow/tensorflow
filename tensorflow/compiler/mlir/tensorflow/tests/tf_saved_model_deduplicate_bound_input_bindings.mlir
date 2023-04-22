// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-saved-model-dedup-bound-input-binding-pass | FileCheck %s

module attributes {tf_saved_model.semantics, tf_saved_model.under_construction} {
  // Test case: Remove duplicate bound_input symbols.
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<42.0> : tensor<f32> } : () -> ()
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "w", type = tensor<f32>, value = dense<43.0> : tensor<f32> } : () -> ()
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "x", type = tensor<f32>, value = dense<44.0> : tensor<f32> } : () -> ()
  // CHECK: func @f
  // CHECK: %arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v}
  // CHECK: %arg1: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @w}
  // CHECK: %arg2: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @x}
  // CHECK-NOT: %arg3
  // CHECK-NOT: %arg4
  func @f(
    %arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v},
    %arg1: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @w},
    %arg2: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v},
    %arg3: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @x},
    %arg4: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v}
  ) attributes {tf_saved_model.exported_names = ["f"]} {
    // CHECK: "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    // CHECK: "tf.ReadVariableOp"(%arg1) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    // CHECK: "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    // CHECK: "tf.ReadVariableOp"(%arg2) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    // CHECK: "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    %val0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    %val1 = "tf.ReadVariableOp"(%arg1) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    %val2 = "tf.ReadVariableOp"(%arg2) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    %val3 = "tf.ReadVariableOp"(%arg3) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    %val4 = "tf.ReadVariableOp"(%arg4) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    return
  }
}
