builtin.func @test(%V__0: tensor<?x?xf32>, %V__1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "tf.Asin"(%V__0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "tf.ClipByValue"(%0, %V__0, %V__0) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "tf.Atan2"(%1, %V__0) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "tf.Sign"(%V__0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = "tf.Selu"(%3) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = "tf.Mul"(%V__1, %4) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = "tf.Cos"(%5) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %7 = "tf.Cos"(%6) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = "tf.Pow"(%V__1, %7) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %9 = "tf.Atan2"(%V__0, %8) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %10 = "tf.LeakyRelu"(%9) {alpha = 0.2 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %11 = "tf.FloorMod"(%10, %V__1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %12 = "tf.Round"(%11) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %13 = "tf.Asinh"(%V__0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %14 = "tf.Xlog1py"(%12, %13) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %15 = "tf.Sub"(%2, %14) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %16 = "tf.LeakyRelu"(%15) {alpha = 0.2 : f32} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %17 = "tf.Softsign"(%16) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %18 = "tf.Rsqrt"(%17) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %19 = "tf.Log"(%18) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %19 : tensor<?x?xf32>
}
