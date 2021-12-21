builtin.func @test(%V__0: tensor<?x?x?xf32>, %V__1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "tf.Atan"(%V__0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = "tf.Erf"(%0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %2 = "tf.Sub"(%1, %V__0) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3 = "tf.Softsign"(%2) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %4 = "tf.Reciprocal"(%3) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %5 = "tf.Acosh"(%V__0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %6 = "tf.Floor"(%5) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %7 = "tf.Asin"(%6) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %8 = "tf.FloorMod"(%7, %V__0) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %9 = "tf.Sinh"(%8) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %10 = "tf.Cos"(%9) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %11 = "tf.Atanh"(%10) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %12 = "tf.Atan2"(%V__0, %V__1) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %13 = "tf.Sin"(%12) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %14 = "tf.Mod"(%11, %13) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %15 = "tf.Atanh"(%14) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %16 = "tf.Sin"(%15) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %17 = "tf.Div"(%4, %16) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %17 : tensor<?x?x?xf32>
}
