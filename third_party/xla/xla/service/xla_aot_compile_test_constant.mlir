module @foo {
  func.func public @main(%arg1: tensor<3 x f64>) -> tensor<3 x f64> {
    %a = mhlo.constant dense<[1.0, 2.0, 3.0]> : tensor<3 x f64>
    %b = mhlo.add %arg1, %a : tensor<3 x f64>
    return %b: tensor<3 x f64>
  }
}