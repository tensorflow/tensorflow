// RUN: tf-opt -tf-to-corert %s | FileCheck %s

module attributes {tf_saved_model.semantics} {

"tf_saved_model.global_tensor"() {is_mutable, sym_name = "y", type = tensor<1x3xf32>, value = dense<[[1.67482901, -0.529208779, -0.803792417]]> : tensor<1x3xf32>} : () -> ()

// CHECK-LABEL: func @basic
func @func_basic(
    %arg0: tensor<3x1xf32> {tf_saved_model.index_path = [0]},
    %arg1: tensor<!tf.resource<tensor<1x3xf32>>> {tf_saved_model.bound_input = @y})
      -> (tensor<3x3xf32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["basic"]} {
  %1 = "tf.ReadVariableOp"(%arg1) {_output_shapes = ["tfshape$dim { size: 1 } dim { size: 3 }"], device = "cpu", dtype = f32} : (tensor<!tf.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>

  // CHECK: {{%.*}} = corert.executeop({{%.*}}) "tf.MatMul"
  // CHECK-SAME: {T = f32, transpose_a = false, transpose_b = false}
  %2 = "tf.MatMul"(%arg0, %1) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], device = "cpu", transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
  return %2 : tensor<3x3xf32>
}

}
