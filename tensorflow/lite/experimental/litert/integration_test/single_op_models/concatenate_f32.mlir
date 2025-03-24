module attributes {tfl.description = "MLIR Converted.", tfl.metadata = {min_runtime_version = "1.5.0\00\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<2x3x2xf32>, %arg1: tensor<2x4x2xf32>, %arg2: tensor<2x1x2xf32>) -> tensor<2x8x2xf32> attributes {tf.entry_function = {inputs = "arg0,arg1,arg2", outputs = "tfl.concatenation"}} {
    %0 = "tfl.concatenation"(%arg0, %arg1, %arg2) <{axis = 1 : i32, fused_activation_function = "NONE"}> : (tensor<2x3x2xf32>, tensor<2x4x2xf32>, tensor<2x1x2xf32>) -> tensor<2x8x2xf32>
    return %0 : tensor<2x8x2xf32>
  }
}
