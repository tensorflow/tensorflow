// RUN: tf-opt %s | tf-opt | FileCheck %s --dump-input=fail

module attributes {tf_saved_model.semantics} {

  // Representation for constants: (immutable) global tensor.
  // CHECK: tf_saved_model.global_tensor
  "tf_saved_model.global_tensor"() {
    tf_saved_model.exported_names = ["some_const"],
    sym_name = "some_constant",
    value = dense<42.0> : tensor<1x64xf32>
  } : () -> ()

  // Representation for variables: mutable global tensor.
  // CHECK: tf_saved_model.global_tensor
  "tf_saved_model.global_tensor"() {
    is_mutable,
    tf_saved_model.exported_names = ["some_var", "some.other.name"],
    sym_name = "some_variable",
    value = dense<42.0> : tensor<1x64xf32>
  } : () -> ()

  // Representation for functions: func's with attributes.
  // CHECK: func @__concrete_function_run_computation
  func @__concrete_function_run_computation(
    %arg0: tensor<f32>,
    %arg1: tensor<f32> {tf_saved_model.bound_input = @some_constant}
  ) -> tensor<f32>
  attributes { tf_saved_model.exported_names = ["some_func"] }
  {
    return %arg0 : tensor<f32>
  }

}
