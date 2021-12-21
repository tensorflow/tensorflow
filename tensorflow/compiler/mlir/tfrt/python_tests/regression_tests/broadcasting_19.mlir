builtin.func @test(%V__0: tensor<?xf32>, %V__1: tensor<?xf32>, %V__2: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "tf.LeakyRelu"(%V__0) {alpha = 0.1 : f32} : (tensor<?xf32>) -> tensor<?xf32>
  %1 = "tf.Sqrt"(%0) : (tensor<?xf32>) -> tensor<?xf32>
  %2 = "tf.Maximum"(%V__0, %1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %3 = "tf.Atan2"(%2, %V__1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %4 = "tf.LeakyRelu"(%3) {alpha = 0.2 : f32} : (tensor<?xf32>) -> tensor<?xf32>
  %5 = "tf.Rint"(%4) : (tensor<?xf32>) -> tensor<?xf32>
  %6 = "tf.Atan"(%5) : (tensor<?xf32>) -> tensor<?xf32>
  %7 = "tf.Rsqrt"(%6) : (tensor<?xf32>) -> tensor<?xf32>
  %8 = "tf.Softplus"(%7) : (tensor<?xf32>) -> tensor<?xf32>
  %9 = "tf.Cosh"(%8) : (tensor<?xf32>) -> tensor<?xf32>
  %10 = "tf.Atan2"(%V__0, %V__0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %11 = "tf.Atan2"(%9, %10) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %12 = "tf.FloorMod"(%11, %V__2) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %12 : tensor<?xf32>
}
