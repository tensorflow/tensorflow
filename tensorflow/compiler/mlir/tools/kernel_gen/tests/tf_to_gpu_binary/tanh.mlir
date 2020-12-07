// RUN: tf_to_gpu_binary --input=%s --output=%t --unroll_factors=4 --tile_sizes=256 --arch=sm_70
func @tanh(%arg0: tensor<?xf32>) -> tensor<?xf32> attributes {tf_entry} {
  %0 = "tf.Tanh"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
