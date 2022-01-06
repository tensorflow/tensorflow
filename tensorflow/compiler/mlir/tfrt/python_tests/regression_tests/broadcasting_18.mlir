builtin.func @test(%V__0: tensor<?x?xf32>, %V__1: tensor<3xi32>, %V__2: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "tf.BroadcastTo"(%V__0, %V__1) : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1 = "tf.Sin"(%0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %2 = "tf.Atan2"(%V__2, %V__2) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3 = "tf.LeakyRelu"(%2) {alpha = 0.3 : f32} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %4 = "tf.Tanh"(%V__2) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %5 = "tf.DivNoNan"(%3, %4) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %6 = "tf.Asinh"(%5) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %7 = "tf.SquaredDifference"(%1, %6) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %8 = "tf.SquaredDifference"(%7, %V__2) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %9 = "tf.Acosh"(%8) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %10 = "tf.Abs"(%V__2) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %11 = "tf.Xlog1py"(%9, %10) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %11 : tensor<?x?x?xf32>
}
