builtin.func @test(%V__0: tensor<?xf32>, %V__1: tensor<?xf32>, %V__2: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "tf.Neg"(%V__0) : (tensor<?xf32>) -> tensor<?xf32>
  %1 = "tf.Div"(%0, %V__1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %2 = "tf.Exp"(%V__2) : (tensor<?xf32>) -> tensor<?xf32>
  %3 = "tf.Pow"(%2, %V__1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %4 = "tf.Reciprocal"(%V__0) : (tensor<?xf32>) -> tensor<?xf32>
  %5 = "tf.Log"(%4) : (tensor<?xf32>) -> tensor<?xf32>
  %6 = "tf.Asin"(%5) : (tensor<?xf32>) -> tensor<?xf32>
  %7 = "tf.SquaredDifference"(%V__1, %6) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %8 = "tf.Rint"(%7) : (tensor<?xf32>) -> tensor<?xf32>
  %9 = "tf.Acos"(%8) : (tensor<?xf32>) -> tensor<?xf32>
  %10 = "tf.Cos"(%9) : (tensor<?xf32>) -> tensor<?xf32>
  %11 = "tf.Maximum"(%3, %10) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %12 = "tf.Acos"(%11) : (tensor<?xf32>) -> tensor<?xf32>
  %13 = "tf.Add"(%1, %12) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %14 = "tf.Relu6"(%13) : (tensor<?xf32>) -> tensor<?xf32>
  %15 = "tf.Asin"(%14) : (tensor<?xf32>) -> tensor<?xf32>
  return %15 : tensor<?xf32>
}
