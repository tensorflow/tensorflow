module {
  func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<2x8xf32> {
    %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }
}