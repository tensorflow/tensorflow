builtin.func @test(%V__0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "tf.Square"(%V__0) : (tensor<?xf32>) -> tensor<?xf32>
  %1 = "tf.Mul"(%V__0, %V__0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %2 = "tf.Asinh"(%1) : (tensor<?xf32>) -> tensor<?xf32>
  %3 = "tf.FloorMod"(%V__0, %V__0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %4 = "tf.Atanh"(%3) : (tensor<?xf32>) -> tensor<?xf32>
  %5 = "tf.Sin"(%4) : (tensor<?xf32>) -> tensor<?xf32>
  %6 = "tf.Exp"(%5) : (tensor<?xf32>) -> tensor<?xf32>
  %7 = "tf.FloorMod"(%2, %6) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %8 = "tf.Atan2"(%0, %7) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %9 = "tf.Relu"(%8) : (tensor<?xf32>) -> tensor<?xf32>
  %10 = "tf.Floor"(%9) : (tensor<?xf32>) -> tensor<?xf32>
  %11 = "tf.Rint"(%10) : (tensor<?xf32>) -> tensor<?xf32>
  %12 = "tf.Xdivy"(%11, %V__0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %13 = "tf.Sin"(%12) : (tensor<?xf32>) -> tensor<?xf32>
  %14 = "tf.Atanh"(%13) : (tensor<?xf32>) -> tensor<?xf32>
  %15 = "tf.Tanh"(%14) : (tensor<?xf32>) -> tensor<?xf32>
  %16 = "tf.Elu"(%V__0) : (tensor<?xf32>) -> tensor<?xf32>
  %17 = "tf.Softplus"(%16) : (tensor<?xf32>) -> tensor<?xf32>
  %18 = "tf.ClipByValue"(%15, %V__0, %17) : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %19 = "tf.Inv"(%18) : (tensor<?xf32>) -> tensor<?xf32>
  return %19 : tensor<?xf32>
}
