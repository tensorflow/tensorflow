builtin.func @test(%V__0: tensor<?xf32>, %V__1: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = "tf.Asinh"(%V__0) : (tensor<?xf32>) -> tensor<?xf32>
  %1 = "tf.BroadcastTo"(%0, %V__1) : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %2 = "tf.Acosh"(%1) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "tf.Sqrt"(%2) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = "tf.Asin"(%3) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = "tf.Round"(%4) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = "tf.Sinh"(%5) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %7 = "tf.Acosh"(%6) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %8 = "tf.Relu6"(%7) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %9 = "tf.Ceil"(%8) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %10 = "tf.Sin"(%9) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %10 : tensor<?x?xf32>
}
