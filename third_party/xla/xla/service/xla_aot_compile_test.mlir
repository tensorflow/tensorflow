module @foo {
  func.func public @main(%arg1: tensor<3 x f64>, %arg2: tensor<3 x f64>) -> tensor<3 x f64> {
    %a = mhlo.add %arg1, %arg2 : tensor<3 x f64>
    return %a: tensor<3 x f64>
  }
}