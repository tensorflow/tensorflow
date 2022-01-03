builtin.func @test(%V__0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "tf.Mod"(%V__0, %V__0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = "tf.Sin"(%0) : (tensor<?xf32>) -> tensor<?xf32>
  %2 = "tf.Selu"(%1) : (tensor<?xf32>) -> tensor<?xf32>
  %3 = "tf.Cos"(%2) : (tensor<?xf32>) -> tensor<?xf32>
  %4 = "tf.Reciprocal"(%3) : (tensor<?xf32>) -> tensor<?xf32>
  %5 = "tf.LeakyRelu"(%V__0) {alpha = 0.1 : f32} : (tensor<?xf32>) -> tensor<?xf32>
  %6 = "tf.Relu"(%5) : (tensor<?xf32>) -> tensor<?xf32>
  %7 = "tf.Tan"(%V__0) : (tensor<?xf32>) -> tensor<?xf32>
  %8 = "tf.DivNoNan"(%6, %7) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %9 = "tf.Square"(%8) : (tensor<?xf32>) -> tensor<?xf32>
  %10 = "tf.Cosh"(%V__0) : (tensor<?xf32>) -> tensor<?xf32>
  %11 = "tf.Softplus"(%10) : (tensor<?xf32>) -> tensor<?xf32>
  %12 = "tf.Mod"(%9, %11) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %13 = "tf.Acos"(%12) : (tensor<?xf32>) -> tensor<?xf32>
  %14 = "tf.Tan"(%13) : (tensor<?xf32>) -> tensor<?xf32>
  %15 = "tf.LeakyRelu"(%14) {alpha = 0.1 : f32} : (tensor<?xf32>) -> tensor<?xf32>
  %16 = "tf.Add"(%4, %15) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %17 = "tf.Asinh"(%16) : (tensor<?xf32>) -> tensor<?xf32>
  %18 = "tf.Softplus"(%17) : (tensor<?xf32>) -> tensor<?xf32>
  %19 = "tf.Div"(%V__0, %V__0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %20 = "tf.FloorMod"(%19, %V__0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %21 = "tf.Div"(%18, %20) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %21 : tensor<?xf32>
}
