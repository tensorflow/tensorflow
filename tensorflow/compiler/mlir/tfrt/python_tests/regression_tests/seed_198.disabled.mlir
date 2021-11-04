builtin.func @test(%V__0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "tf.Sub"(%V__0, %V__0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = "tf.Sqrt"(%0) : (tensor<?xf32>) -> tensor<?xf32>
  %2 = "tf.Pow"(%V__0, %V__0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %3 = "tf.Floor"(%2) : (tensor<?xf32>) -> tensor<?xf32>
  %4 = "tf.Sinh"(%3) : (tensor<?xf32>) -> tensor<?xf32>
  %5 = "tf.Softsign"(%4) : (tensor<?xf32>) -> tensor<?xf32>
  %6 = "tf.Asin"(%5) : (tensor<?xf32>) -> tensor<?xf32>
  %7 = "tf.Floor"(%6) : (tensor<?xf32>) -> tensor<?xf32>
  %8 = "tf.Rint"(%7) : (tensor<?xf32>) -> tensor<?xf32>
  %9 = "tf.Atanh"(%8) : (tensor<?xf32>) -> tensor<?xf32>
  %10 = "tf.Erf"(%9) : (tensor<?xf32>) -> tensor<?xf32>
  %11 = "tf.Asin"(%10) : (tensor<?xf32>) -> tensor<?xf32>
  %12 = "tf.Floor"(%11) : (tensor<?xf32>) -> tensor<?xf32>
  %13 = "tf.FloorDiv"(%1, %12) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %13 : tensor<?xf32>
}
