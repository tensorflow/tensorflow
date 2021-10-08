builtin.func @test(%V__0: tensor<?x?xf32>, %V__1: tensor<?xf32>) -> tensor<?x?xf32> {
  %0 = "tf.Reciprocal"(%V__0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "tf.Round"(%0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "tf.Acos"(%1) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "tf.Atan"(%V__1) : (tensor<?xf32>) -> tensor<?xf32>
  %4 = "tf.BiasAdd"(%V__0, %3) {data_format = "NHWC"} : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
  %5 = "tf.Square"(%4) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = "tf.Div"(%5, %V__0) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %7 = "tf.FloorMod"(%6, %V__0) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = "tf.Floor"(%V__0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %9 = "tf.Rint"(%V__0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %10 = "tf.Xlog1py"(%8, %9) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %11 = "tf.ClipByValue"(%2, %7, %10) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %12 = "tf.Rint"(%11) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %13 = "tf.Round"(%12) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %14 = "tf.Softplus"(%V__0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %15 = "tf.Neg"(%V__0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %16 = "tf.Asin"(%15) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %17 = "tf.Pow"(%14, %16) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %18 = "tf.Cos"(%17) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %19 = "tf.Sign"(%18) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %20 = "tf.MulNoNan"(%13, %19) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %20 : tensor<?x?xf32>
}
