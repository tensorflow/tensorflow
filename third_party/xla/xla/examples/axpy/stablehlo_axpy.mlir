func.func @main(
  %alpha: tensor<f32>, %x: tensor<4xf32>, %y: tensor<4xf32>
) -> tensor<4xf32> {
  %0 = stablehlo.broadcast_in_dim %alpha, dims = []
    : (tensor<f32>) -> tensor<4xf32>
  %1 = stablehlo.multiply %0, %x : tensor<4xf32>
  %2 = stablehlo.add %1, %y : tensor<4xf32>
  func.return %2: tensor<4xf32>
}
