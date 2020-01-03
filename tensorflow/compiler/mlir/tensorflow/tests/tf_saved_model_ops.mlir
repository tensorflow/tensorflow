// RUN: tf-opt %s | tf-opt | FileCheck %s --dump-input=fail

module attributes {tf_saved_model.semantics} {

  // Representation for constants: (immutable) global tensor.
  // CHECK: tf_saved_model.global_tensor
  "tf_saved_model.global_tensor"() {
    tf_saved_model.exported_names = ["some_const"],
    sym_name = "some_constant",
    type = tensor<1x64xf32>,
    value = dense<42.0> : tensor<1x64xf32>
  } : () -> ()

  // Representation for variables: mutable global tensor.
  // CHECK: tf_saved_model.global_tensor
  "tf_saved_model.global_tensor"() {
    is_mutable,
    tf_saved_model.exported_names = ["some_var", "some.other.name"],
    sym_name = "some_variable",
    type = tensor<?x64xf32>,
    value = dense<42.0> : tensor<1x64xf32>
  } : () -> ()

  // Representation for functions: func's with attributes.
  // CHECK: func @__concrete_function_run_computation
  func @__concrete_function_run_computation(
    %arg0: tensor<f32> {tf_saved_model.index_path = [0, "foo"]},
    %arg1: tensor<f32> {tf_saved_model.bound_input = @some_constant}
  ) -> (
    tensor<f32> {tf_saved_model.index_path = [0, "bar"]}
  ) attributes { tf_saved_model.exported_names = ["some_func"] }
  {
    "some_dialect.some_call"() {callee = @f} : () -> ()
    return %arg0 : tensor<f32>
  }

  func @f() {
    return
  }

}
