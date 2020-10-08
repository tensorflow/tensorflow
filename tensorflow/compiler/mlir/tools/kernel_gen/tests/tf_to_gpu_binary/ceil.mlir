// RUN: tf_to_gpu_binary --input=%s --output=%t --same_shape=0,1 --unroll_factors=4 --tile_sizes=256 --arch=sm_70
func @ceil(%arg0: tensor<?xf64>) -> tensor<?xf64> {
  %0 = "tf.Ceil"(%arg0) { }
    : (tensor<?xf64>) -> tensor<?xf64>
  return %0 : tensor<?xf64>
}
