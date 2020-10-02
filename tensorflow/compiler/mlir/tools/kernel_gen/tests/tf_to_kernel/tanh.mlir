// RUN: tf_to_kernel --input=%s --output=%t --same_shape=0,1 --unroll_factors=4 --tile_sizes=256 --arch=70,75

func @tanh(%arg: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Tanh"(%arg) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
