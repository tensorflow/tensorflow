// RUN: tf-opt "-tf-saved-model-freeze-global-tensors=allow-mutable-tensors=true" -split-input-file %s | FileCheck %s

module attributes {tf_saved_model.semantics} {

  // Test case: Do not fail if the tensor is mutable but allow_mutable_tensors is true

  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  func @f(%arg0: tensor<!tf.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    // CHECK: "tf.ReadVariableOp"
    %0 = "tf.ReadVariableOp"(%arg0) {device = ""} : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    return
  }

}
