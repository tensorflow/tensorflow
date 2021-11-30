module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 398 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "y", type = tensor<3x1xi32>, value = dense<[[1], [2], [3]]> : tensor<3x1xi32>} : () -> ()
  "tf_saved_model.asset"() {sym_name = "z", filename = "file"} : () -> ()
  func @serving_default(
      %arg0: tensor<1x3xi32> {tf_saved_model.index_path = ["x"]},
      %arg1: tensor<!tf_type.resource<tensor<3x1xi32>>> {tf_saved_model.bound_input = @y},
      %arg2: tensor<!tf_type.string> {tf_saved_model.bound_input = @z}
    ) -> (tensor<1x1xi32> {tf_saved_model.index_path = ["r"]})
      attributes {
        tf.entry_function = {control_outputs = "", inputs = "input:0", outputs = "result:0"},
        tf_saved_model.exported_names = ["serving_default"]
    }
  {
    %0 = "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<!tf_type.resource<tensor<3x1xi32>>>) -> tensor<3x1xi32>
    %1 = "tf.MatMul"(%arg0, %0) {device = "", transpose_a = false, transpose_b = false} : (tensor<1x3xi32>, tensor<3x1xi32>) -> tensor<1x1xi32>
    return %1 : tensor<1x1xi32>
  }
  func @predict(
    ) -> (tensor<0x!tf_type.string> {tf_saved_model.index_path = ["r"]})
      attributes {
        tf.entry_function = {control_outputs = "", inputs = "input:0", outputs = "result:0"},
        tf_saved_model.exported_names = ["predict"]
    }
  {
    %0 = "tf.Const"() {dtype = !tf_type.string, value = dense<[]> : tensor<0x!tf_type.string>} : () -> tensor<0x!tf_type.string>
    return %0 : tensor<0x!tf_type.string>
  }
}
