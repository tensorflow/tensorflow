//RUN: tf_tfl_translate --enable-stablehlo-conversion --input-mlir %s -o /tmp/temp.stablehlo; [ -f /tmp/temp.stablehlo ]

module {
func.func @main(%arg0: tensor<4x68x68x3xf32>, %arg1: tensor<5x5x3x8xf32>) -> tensor<4x64x64x8xf32> {
  %0 = "tf.Conv2D"(%arg0, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x3xf32>, tensor<5x5x3x8xf32>) -> tensor<4x64x64x8xf32>
  func.return %0 : tensor<4x64x64x8xf32>
}
}