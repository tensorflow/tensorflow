builtin.func @test(%V__0: tensor<?x?xf32>, %V__1: tensor<?xf32>, %V__2: tensor<?xf32>) -> tensor<?x?xf32> {
  %0 = "tf.Atanh"(%V__0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "tf.Abs"(%V__2) : (tensor<?xf32>) -> tensor<?xf32>
  %2 = "tf.Div"(%1, %V__1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %3 = "tf.Erf"(%2) : (tensor<?xf32>) -> tensor<?xf32>
  %4 = "tf.Exp"(%3) : (tensor<?xf32>) -> tensor<?xf32>
  %5 = "tf.Softsign"(%V__1) : (tensor<?xf32>) -> tensor<?xf32>
  %6 = "tf.Pow"(%4, %5) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %7 = "tf.Reciprocal"(%6) : (tensor<?xf32>) -> tensor<?xf32>
  %8 = "tf.Softsign"(%7) : (tensor<?xf32>) -> tensor<?xf32>
  %9 = "tf.Sqrt"(%8) : (tensor<?xf32>) -> tensor<?xf32>
  %10 = "tf.Sub"(%V__1, %9) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %11 = "tf.Inv"(%V__2) : (tensor<?xf32>) -> tensor<?xf32>
  %12 = "tf.Xlog1py"(%10, %11) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %13 = "tf.BiasAdd"(%0, %12) {data_format = "NHWC"} : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
  %14 = "tf.Rint"(%13) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %15 = "tf.Softplus"(%V__0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %16 = "tf.Add"(%14, %15) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %17 = "tf.Asinh"(%16) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %18 = "tf.Div"(%17, %V__0) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %18 : tensor<?x?xf32>
}
