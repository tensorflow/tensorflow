builtin.func @test(%V__0: tensor<?x?x?xf32>, %V__1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "tf.Exp"(%V__0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = "tf.Cosh"(%V__0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %2 = "tf.Xdivy"(%V__0, %1) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3 = "tf.Floor"(%2) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %4 = "tf.SquaredDifference"(%V__1, %3) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %5 = "tf.Atanh"(%4) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %6 = "tf.Sin"(%5) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %7 = "tf.Atan2"(%0, %6) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %8 = "tf.LeakyRelu"(%7) {alpha = 0.2 : f32} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %9 = "tf.LeakyRelu"(%8) {alpha = 0.1 : f32} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %10 = "tf.SquaredDifference"(%9, %V__0) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %11 = "tf.Rint"(%10) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %12 = "tf.FloorDiv"(%V__1, %V__0) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %13 = "tf.Add"(%11, %12) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %14 = "tf.Neg"(%13) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %14 : tensor<?x?x?xf32>
}
