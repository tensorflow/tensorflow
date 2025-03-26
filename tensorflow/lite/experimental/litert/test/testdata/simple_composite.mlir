module attributes {tfl.description = "MLIR Converted.", tfl.metadata = {min_runtime_version = "1.5.0\00\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32, tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<2x2xf32> { tf_saved_model.index_path = ["arg0"] }, %arg1: tensor<2x2xf32> { tf_saved_model.index_path = ["arg1"]}) -> (tensor<2x2xf32> {tf_saved_model.index_path = ["output"] }) attributes {tf.entry_function = {inputs = "arg0,arg1", outputs = "output"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = stablehlo.composite "odml.npu_call" %arg0, %arg1 {decomposition = @decomp} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
  func.func private @decomp(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}

