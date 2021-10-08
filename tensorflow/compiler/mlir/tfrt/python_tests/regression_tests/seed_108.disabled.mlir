builtin.func @test(%V__0: tensor<?xf32>, %V__1: tensor<?xf32>, %V__2: tensor<?xf32>, %V__3: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "tf.Rint"(%V__0) : (tensor<?xf32>) -> tensor<?xf32>
  %1 = "tf.RealDiv"(%V__0, %0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %2 = "tf.Relu"(%1) : (tensor<?xf32>) -> tensor<?xf32>
  %3 = "tf.Erf"(%2) : (tensor<?xf32>) -> tensor<?xf32>
  %4 = "tf.Erf"(%3) : (tensor<?xf32>) -> tensor<?xf32>
  %5 = "tf.Reciprocal"(%V__2) : (tensor<?xf32>) -> tensor<?xf32>
  %6 = "tf.Xlogy"(%V__1, %5) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %7 = "tf.Rint"(%V__1) : (tensor<?xf32>) -> tensor<?xf32>
  %8 = "tf.Abs"(%7) : (tensor<?xf32>) -> tensor<?xf32>
  %9 = "tf.Relu6"(%8) : (tensor<?xf32>) -> tensor<?xf32>
  %10 = "tf.Xlog1py"(%6, %9) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %11 = "tf.Erf"(%V__3) : (tensor<?xf32>) -> tensor<?xf32>
  %12 = "tf.ClipByValue"(%4, %10, %11) : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %13 = "tf.Round"(%12) : (tensor<?xf32>) -> tensor<?xf32>
  return %13 : tensor<?xf32>
}
