// RUN: tf-opt -tf-saved-model-optimize-global-tensors -split-input-file %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Immutability.
//===----------------------------------------------------------------------===//

module attributes {tf_saved_model.semantics} {

  // Test case: Basic test of marking immutable.

  // CHECK: "tf_saved_model.global_tensor"() {
  // CHECK-NOT: is_mutable
  // CHECK-SAME: } : () -> ()
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<42.> : tensor<f32> } : () -> ()

  func @f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v}) -> (tensor<f32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %val = "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    return %val : tensor<f32>
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  // Test case: Don't mark immutable if the variable is mutated.

  // CHECK: "tf_saved_model.global_tensor"() {
  // CHECK-SAME: is_mutable
  // CHECK-SAME: } : () -> ()
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<42.> : tensor<f32> } : () -> ()

  func @f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %c0 = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
    "tf.AssignVariableOp"(%arg0, %c0) : (tensor<!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
    return
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  // Test case: Don't mark immutable if the variable is exported.

  // CHECK: "tf_saved_model.global_tensor"() {
  // CHECK: is_mutable
  // CHECK-SAME: } : () -> ()
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", tf_saved_model.exported_names = ["v"], type = tensor<f32>, value = dense<42.> : tensor<f32> } : () -> ()

  func @f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v}) -> (tensor<f32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %val = "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    return %val : tensor<f32>
  }

}


// -----

module attributes {tf_saved_model.semantics} {

  // Test case: Check that a non-bound input is left unchanged.

  // CHECK: func @g
  func @g(%arg0: tensor<f32> {tf_saved_model.index_path = [0]}) -> (tensor<f32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["g"]} {
    // CHECK: return %arg0
    return %arg0 : tensor<f32>
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  // Test case: Check that no change is made for a global tensor that is already
  // immutable.

  "tf_saved_model.global_tensor"() { sym_name = "c", type = tensor<f32>, value = dense<42.> : tensor<f32> } : () -> ()

  // CHECK: func @h(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @c})
  func @h(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @c})
  attributes {tf_saved_model.exported_names = ["h"]} {
    %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    return
  }

}

// -----

//===----------------------------------------------------------------------===//
// Erasing unused global tensors.
//===----------------------------------------------------------------------===//

module attributes {tf_saved_model.semantics} {

  // Test case: Check that an exported global tensor that isn't bound to an
  // argument is not erased.

  "tf_saved_model.global_tensor"() { sym_name = "exported_unbound", tf_saved_model.exported_names = ["exported_unbound"], type = tensor<f32>, value = dense<42.> : tensor<f32> } : () -> ()
  // CHECK: sym_name = "exported_unbound"

  // Test case: Check that a global tensor that isn't even bound to an argument
  // is erased.

  "tf_saved_model.global_tensor"() { sym_name = "unexported_unbound", type = tensor<f32>, value = dense<42.> : tensor<f32> } : () -> ()
  // CHECK-NOT: sym_name = "unexported_unbound"

}

// -----

//===----------------------------------------------------------------------===//
// Erasing unused bound inputs.
//===----------------------------------------------------------------------===//

module attributes {tf_saved_model.semantics} {

  // We erase the argument that this global tensor is bound to, so we delete
  // the global tensor too.
  // CHECK-NOT: tf_saved_model.global_tensor
  "tf_saved_model.global_tensor"() { sym_name = "c", type = tensor<f32>, value = dense<42.> : tensor<f32> } : () -> ()

  // CHECK: func @f()
  func @f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @c})
  attributes {tf_saved_model.exported_names = ["f"]} {
    return
  }

}

// -----

// Test running the pass on a module that does not have
// tf_saved_model.semantics.
module {}

// -----

// Test use as an input in unhandled op
module attributes {tf_saved_model.semantics} {

  // CHECK: "tf_saved_model.global_tensor"() {
  // CHECK-SAME: is_mutable
  // CHECK-SAME: } : () -> ()
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<42.> : tensor<f32> } : () -> ()

  func @f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    "tf.unhandled_op"(%arg0) : (tensor<!tf.resource<tensor<f32>>>) -> ()
    return
  }
}


// -----

// Test use as a region capture in an unhandled op
module attributes {tf_saved_model.semantics} {

  // CHECK: "tf_saved_model.global_tensor"() {
  // CHECK-SAME: is_mutable
  // CHECK-SAME: } : () -> ()
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<42.> : tensor<f32> } : () -> ()

  func @f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    "tf.unhandled"() ({
      %val = "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
      "tf.unhandled_terminator"() : () -> ()
    }) : () -> ()
    return
  }
}

// -----

// Test use as region capture as well as input in an unhandled op
// to the unhandled op.
module attributes {tf_saved_model.semantics} {

  // CHECK: "tf_saved_model.global_tensor"() {
  // CHECK-SAME: is_mutable
  // CHECK-SAME: } : () -> ()
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<42.> : tensor<f32> } : () -> ()

  // CHECK: "tf_saved_model.global_tensor"() {
  // CHECK-SAME: is_mutable
  // CHECK-SAME: } : () -> ()
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "u", type = tensor<f32>, value = dense<22.> : tensor<f32> } : () -> ()

  func @f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v}, %arg1: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @u})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %0 = "tf.unhandled"(%arg0) ({
      %val = "tf.ReadVariableOp"(%arg1) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
      "tf.unhandled_terminator"() : () -> ()
    }) : (tensor<!tf.resource<tensor<f32>>>) -> (tensor<!tf.resource<tensor<f32>>>)
    return
  }
}

// -----

// Test multiple global tensors uses as operands for an unhandled op.
module attributes {tf_saved_model.semantics} {

  // CHECK: "tf_saved_model.global_tensor"() {
  // CHECK-SAME: is_mutable
  // CHECK-SAME: } : () -> ()
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<42.> : tensor<f32> } : () -> ()

  // CHECK: "tf_saved_model.global_tensor"() {
  // CHECK-SAME: is_mutable
  // CHECK-SAME: } : () -> ()
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "u", type = tensor<f32>, value = dense<22.> : tensor<f32> } : () -> ()

  func @f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v}, %arg1: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @u})
  attributes {tf_saved_model.exported_names = ["f"]} {
    "tf.unhandled"(%arg0, %arg1) : (tensor<!tf.resource<tensor<f32>>>, tensor<!tf.resource<tensor<f32>>>) -> ()
    return
  }
}

// -----

// Test presence of tf_saved_model.asset's as bound inputs.
// It should not crash.
module attributes {tf_saved_model.semantics}  {

  "tf_saved_model.session_initializer"() {initializers = [@legacy_init_op]} : () -> ()
  "tf_saved_model.asset"() {filename = "assets/foo.txt", sym_name = "asset"} : () -> ()
  // CHECK: @legacy_init_op
  func @legacy_init_op(%arg0: tensor<!tf.string> {tf_saved_model.bound_input = @asset}) attributes {tf_saved_model.exported_names = ["f"]} {
    %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf.string, shared_name = "hash_table_shared_name", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf.resource>
    "tf.InitializeTableFromTextFileV2"(%0, %arg0) {delimiter = "\09", device = "", key_index = -2 : i64, value_index = -1 : i64, vocab_size = 205 : i64} : (tensor<!tf.resource>, tensor<!tf.string>) -> ()
    return
  }
}
