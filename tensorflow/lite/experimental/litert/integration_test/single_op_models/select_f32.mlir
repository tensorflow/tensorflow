module attributes {tfl.description = "MLIR Converted.", tfl.metadata = {min_runtime_version = "1.14.0\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<2x2xi1>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {tf.entry_function = {inputs = "arg0,arg1,arg2", outputs = "tfl.select"}} {
    %0 = "tfl.select"(%arg0, %arg1, %arg2) : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
