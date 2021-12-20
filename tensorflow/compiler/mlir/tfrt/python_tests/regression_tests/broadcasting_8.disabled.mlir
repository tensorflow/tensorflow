builtin.func @test(%V__0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "tf.Sqrt"(%V__0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "tf.Mul"(%V__0, %V__0) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "tf.FloorDiv"(%1, %V__0) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "tf.Pow"(%0, %2) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = "tf.Cos"(%3) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = "tf.Round"(%4) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = "tf.Rsqrt"(%V__0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %7 = "tf.Abs"(%6) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = "tf.FloorDiv"(%7, %V__0) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %9 = "tf.Div"(%5, %8) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %10 = "tf.Asinh"(%9) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %11 = "tf.Atanh"(%10) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %12 = "tf.LeakyRelu"(%11) {alpha = 0.3 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %13 = "tf.Selu"(%12) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %14 = "tf.Ceil"(%13) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %15 = "tf.Maximum"(%V__0, %V__0) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %16 = "tf.Elu"(%15) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %17 = "tf.Floor"(%16) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %18 = "tf.Xlogy"(%14, %17) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %18 : tensor<?x?xf32>
}
