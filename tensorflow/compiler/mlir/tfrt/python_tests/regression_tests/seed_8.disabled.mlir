builtin.func @test(%V__0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "tf.Mod"(%V__0, %V__0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = "tf.MulNoNan"(%V__0, %V__0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %2 = "tf.Acosh"(%1) : (tensor<?xf32>) -> tensor<?xf32>
  %3 = "tf.Cos"(%2) : (tensor<?xf32>) -> tensor<?xf32>
  %4 = "tf.Atan"(%3) : (tensor<?xf32>) -> tensor<?xf32>
  %5 = "tf.Mul"(%V__0, %4) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %6 = "tf.Atan"(%5) : (tensor<?xf32>) -> tensor<?xf32>
  %7 = "tf.Round"(%6) : (tensor<?xf32>) -> tensor<?xf32>
  %8 = "tf.Softsign"(%7) : (tensor<?xf32>) -> tensor<?xf32>
  %9 = "tf.SquaredDifference"(%V__0, %8) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %10 = "tf.Sign"(%9) : (tensor<?xf32>) -> tensor<?xf32>
  %11 = "tf.Div"(%0, %10) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %11 : tensor<?xf32>
}
