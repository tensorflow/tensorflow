module attributes {tfl.description = "MLIR Converted.", tfl.metadata = {min_runtime_version = "1.10.0\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<256xf32>) -> tensor<256xf32> attributes {tf.entry_function = {inputs = "arg0", outputs = "tfl.rsqrt"}} {
    %0 = "tfl.rsqrt"(%arg0) : (tensor<256xf32>) -> tensor<256xf32>
    return %0 : tensor<256xf32>
  }
}
