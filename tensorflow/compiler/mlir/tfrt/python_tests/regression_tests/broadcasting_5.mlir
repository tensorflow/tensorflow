builtin.func @test(%V__0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "tf.Asinh"(%V__0) : (tensor<?xf32>) -> tensor<?xf32>
  %1 = "tf.Elu"(%0) : (tensor<?xf32>) -> tensor<?xf32>
  %2 = "tf.Square"(%1) : (tensor<?xf32>) -> tensor<?xf32>
  %3 = "tf.Tanh"(%2) : (tensor<?xf32>) -> tensor<?xf32>
  %4 = "tf.Cosh"(%3) : (tensor<?xf32>) -> tensor<?xf32>
  %5 = "tf.Round"(%4) : (tensor<?xf32>) -> tensor<?xf32>
  %6 = "tf.Round"(%5) : (tensor<?xf32>) -> tensor<?xf32>
  %7 = "tf.Sign"(%6) : (tensor<?xf32>) -> tensor<?xf32>
  %8 = "tf.Acos"(%V__0) : (tensor<?xf32>) -> tensor<?xf32>
  %9 = "tf.Reciprocal"(%8) : (tensor<?xf32>) -> tensor<?xf32>
  %10 = "tf.Div"(%V__0, %9) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %11 = "tf.Round"(%10) : (tensor<?xf32>) -> tensor<?xf32>
  %12 = "tf.Rint"(%11) : (tensor<?xf32>) -> tensor<?xf32>
  %13 = "tf.DivNoNan"(%7, %12) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %14 = "tf.Cosh"(%13) : (tensor<?xf32>) -> tensor<?xf32>
  return %14 : tensor<?xf32>
}
