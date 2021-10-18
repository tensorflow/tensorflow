builtin.func @test(%V__0: tensor<?x?xf32>, %V__1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "tf.Atan"(%V__0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "tf.Log"(%V__0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "tf.FloorDiv"(%V__0, %1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "tf.DivNoNan"(%0, %2) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = "tf.Exp"(%3) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = "tf.Sign"(%4) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = "tf.Floor"(%5) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %7 = "tf.Asinh"(%6) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = "tf.Sqrt"(%7) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %9 = "tf.Sigmoid"(%8) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %10 = "tf.Round"(%9) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %11 = "tf.Softsign"(%10) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %12 = "tf.Inv"(%11) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %13 = "tf.ClipByValue"(%V__0, %V__0, %V__0) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %14 = "tf.Selu"(%13) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %15 = "tf.Mul"(%V__0, %14) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %16 = "tf.Cosh"(%15) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %17 = "tf.Abs"(%V__1) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %18 = "tf.Pow"(%V__1, %17) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %19 = "tf.FloorDiv"(%16, %18) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %20 = "tf.AddV2"(%12, %19) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %20 : tensor<?x?xf32>
}
