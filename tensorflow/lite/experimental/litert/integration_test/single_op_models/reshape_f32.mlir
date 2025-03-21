module attributes {tfl.description = "MLIR Converted.", tfl.metadata = {min_runtime_version = "1.5.0\00\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<3x4xf32>) -> tensor<4x3xf32> attributes {tf.entry_function = {inputs = "arg0", outputs = "tfl.reshape"}} {
    %0 = "tfl.pseudo_const"() <{value = dense<[4, 3]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1 = "tfl.reshape"(%arg0, %0) : (tensor<3x4xf32>, tensor<2xi32>) -> tensor<4x3xf32>
    return %1 : tensor<4x3xf32>
  }
}
