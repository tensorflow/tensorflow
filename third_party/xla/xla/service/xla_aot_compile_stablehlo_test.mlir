module @axpy {
  func.func public @main(%alpha: tensor<f32>, %x: tensor<4 x f32>, %y: tensor<4 x f32>) -> tensor<4 x f32> {
    %a = "stablehlo.broadcast_in_dim" (%alpha) {
      broadcast_dimensions = dense<[]> : tensor<0 x i64>
    } : (tensor<f32>) -> tensor<4 x f32>
    %ax = stablehlo.multiply %a, %x : tensor<4 x f32>
    %result = stablehlo.add %ax, %y : tensor<4 x f32>
    return %result: tensor<4 x f32>
  }
}
