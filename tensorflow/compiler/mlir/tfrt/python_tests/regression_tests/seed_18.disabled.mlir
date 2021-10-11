builtin.func @test(%V__0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "tf.Elu"(%V__0) : (tensor<?xf32>) -> tensor<?xf32>
  %1 = "tf.Tanh"(%0) : (tensor<?xf32>) -> tensor<?xf32>
  %2 = "tf.LeakyRelu"(%1) {alpha = 0.1 : f32} : (tensor<?xf32>) -> tensor<?xf32>
  %3 = "tf.Rsqrt"(%2) : (tensor<?xf32>) -> tensor<?xf32>
  %4 = "tf.Neg"(%3) : (tensor<?xf32>) -> tensor<?xf32>
  %5 = "tf.Erf"(%4) : (tensor<?xf32>) -> tensor<?xf32>
  %6 = "tf.Erf"(%5) : (tensor<?xf32>) -> tensor<?xf32>
  %7 = "tf.Relu6"(%V__0) : (tensor<?xf32>) -> tensor<?xf32>
  %8 = "tf.Erf"(%7) : (tensor<?xf32>) -> tensor<?xf32>
  %9 = "tf.Softsign"(%8) : (tensor<?xf32>) -> tensor<?xf32>
  %10 = "tf.Exp"(%9) : (tensor<?xf32>) -> tensor<?xf32>
  %11 = "tf.Ceil"(%V__0) : (tensor<?xf32>) -> tensor<?xf32>
  %12 = "tf.Inv"(%11) : (tensor<?xf32>) -> tensor<?xf32>
  %13 = "tf.Minimum"(%10, %12) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %14 = "tf.Xlog1py"(%6, %13) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %15 = "tf.Atanh"(%14) : (tensor<?xf32>) -> tensor<?xf32>
  %16 = "tf.Log"(%15) : (tensor<?xf32>) -> tensor<?xf32>
  %17 = "tf.Elu"(%16) : (tensor<?xf32>) -> tensor<?xf32>
  %18 = "tf.Erf"(%17) : (tensor<?xf32>) -> tensor<?xf32>
  return %18 : tensor<?xf32>
}
