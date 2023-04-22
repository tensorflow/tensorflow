// RUN: tf_to_kernel --input=%s --output=%t --unroll_factors=4 --tile_sizes=256 --arch=sm_70

func @AddV2(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>)
    -> tensor<*xf32> attributes {tf_entry, llvm.emit_c_interface} {
  %0 = "tf.AddV2"(%arg0, %arg1) {T = f32, device = ""}
    : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
