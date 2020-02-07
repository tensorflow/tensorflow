// RUN: tf-opt -tf-saved-model-inline-global-tensors -split-input-file %s | FileCheck %s --dump-input=fail

module attributes {tf_saved_model.semantics} {

  // Test case: Simple case of inlining.

  // CHECK-NOT: tf_saved_model.global_tensor
  "tf_saved_model.global_tensor"() { sym_name = "c", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  // CHECK: func @f()
  func @f(%arg0: tensor<f32> {tf_saved_model.bound_input = @c})
  attributes {tf_saved_model.exported_names = ["f"]} {
    // CHECK: "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>}
    return
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  // Test case: Do not inline mutable global tensors.

  // CHECK: tf_saved_model.global_tensor
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  // CHECK: func @f(%arg0: tensor<*x!tf.resource> {tf_saved_model.bound_input = @v})
  func @f(%arg0: tensor<*x!tf.resource> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    // CHECK-NOT: tf.Const
    return
  }

}

// -----

module attributes {tf_saved_model.semantics} {

  // Test case: Sanity check handling of non-bound inputs.
  // The pass shouldn't do anything in this case.

  // CHECK: func @f(%arg0: tensor<f32> {tf_saved_model.index_path = [0]})
  func @f(%arg0: tensor<f32> {tf_saved_model.index_path = [0]})
  attributes {tf_saved_model.exported_names = ["f"]} {
    // CHECK-NOT: tf.Const
    return
  }

}

// TODO: have an arg that isn't a bound input.
