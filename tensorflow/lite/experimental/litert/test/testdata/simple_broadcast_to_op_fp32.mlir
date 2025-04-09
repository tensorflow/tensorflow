module {
  func.func @main(%arg0: tensor<1x1x1x196xf32>, %arg1: tensor<4xi64>) -> tensor<1x1x196x196xf32> {
    %0 = "tfl.broadcast_to"(%arg0, %arg1) : (tensor<1x1x1x196xf32>, tensor<4xi64>) -> tensor<1x1x196x196xf32>
    return %0 : tensor<1x1x196x196xf32>
  }
}