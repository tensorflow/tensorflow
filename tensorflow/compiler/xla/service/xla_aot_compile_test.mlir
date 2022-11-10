module @foo {
  func.func public @main(%arg1: tensor<f64>) -> tensor<f64> {
    return %arg1 : tensor<f64>
  }
}