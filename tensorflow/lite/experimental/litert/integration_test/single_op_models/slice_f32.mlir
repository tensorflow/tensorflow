module attributes {tfl.description = "MLIR Converted.", tfl.metadata = {min_runtime_version = "1.14.0\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<3x4xf32>) -> tensor<2x2xf32> attributes {tf.entry_function = {inputs = "arg0", outputs = "tfl.slice"}} {
    %0 = "tfl.pseudo_const"() <{value = dense<[1, 2]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1 = "tfl.pseudo_const"() <{value = dense<2> : tensor<2xi32>}> : () -> tensor<2xi32>
    %2 = "tfl.slice"(%arg0, %0, %1) : (tensor<3x4xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xf32>
    return %2 : tensor<2x2xf32>
  }
}
