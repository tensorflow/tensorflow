// RUN: tf_to_gpu_binary --input=%s --output=%t --same_shape=0,1 --unroll_factors=4 --tile_sizes=256 --arch=sm_70
func @tanh(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "tf.Tanh"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
