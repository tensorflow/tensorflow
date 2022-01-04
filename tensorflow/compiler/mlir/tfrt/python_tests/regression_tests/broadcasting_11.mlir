builtin.func @test(%V__0: tensor<?x?xf32>, %V__1: tensor<3xi32>, %V__2: tensor<?x?x?xf32>, %V__3: tensor<?xf32>) -> tensor<?x?x?xf32> {
  %0 = "tf.BroadcastTo"(%V__0, %V__1) : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1 = "tf.Abs"(%0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %2 = "tf.Ceil"(%1) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3 = "tf.Abs"(%2) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %4 = "tf.Log"(%V__2) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %5 = "tf.Square"(%4) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %6 = "tf.Exp"(%5) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %7 = "tf.Acosh"(%6) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %8 = "tf.Asinh"(%7) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %9 = "tf.LeakyRelu"(%V__3) {alpha = 0.1 : f32} : (tensor<?xf32>) -> tensor<?xf32>
  %10 = "tf.BiasAdd"(%V__0, %9) {data_format = "NHWC"} : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
  %11 = "tf.BroadcastTo"(%10, %V__1) : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %12 = "tf.Add"(%11, %V__2) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %13 = "tf.Erf"(%12) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %14 = "tf.MulNoNan"(%8, %13) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %15 = "tf.Ceil"(%14) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %16 = "tf.Mul"(%15, %V__2) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %17 = "tf.Rint"(%16) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %18 = "tf.Add"(%3, %17) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %19 = "tf.Relu6"(%18) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %19 : tensor<?x?x?xf32>
}
