module attributes {tfl.description = "MLIR Converted.", tfl.metadata = {min_runtime_version = "1.5.0\00\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<2x3x4x5x6x7x8xf32>) -> tensor<8x7x6x5x4x3x2xf32> attributes {tf.entry_function = {inputs = "arg0", outputs = "tfl.reshape"}} {
    %0 = "tfl.pseudo_const"() <{value = dense<[8, 7, 6, 5, 4, 3, 2]> : tensor<7xi32>}> : () -> tensor<7xi32>
    %1 = "tfl.reshape"(%arg0, %0) : (tensor<2x3x4x5x6x7x8xf32>, tensor<7xi32>) -> tensor<8x7x6x5x4x3x2xf32>
    return %1 : tensor<8x7x6x5x4x3x2xf32>
  }
}
