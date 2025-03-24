module attributes {tfl.description = "MLIR Converted.", tfl.metadata = {min_runtime_version = "1.14.0\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>) -> tensor<256x256xi1> attributes {tf.entry_function = {inputs = "arg0,arg1", outputs = "tfl.less"}} {
    %0 = tfl.less(%arg0, %arg1) : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xi1>
    return %0 : tensor<256x256xi1>
  }
}
