module attributes {tfl.description = "MLIR Converted.", tfl.metadata = {min_runtime_version = "1.6.0\00\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>) -> tensor<256x256xf32> attributes {tf.entry_function = {inputs = "arg0,arg1", outputs = "tfl.sub"}} {
    %0 = tfl.sub %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
}
