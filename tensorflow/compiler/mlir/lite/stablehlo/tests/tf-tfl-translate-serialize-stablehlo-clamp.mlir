//RUN: tf_tfl_translate --enable-stablehlo-conversion --input-mlir %s -o /tmp/temp.stablehlo; [ -f /tmp/temp.stablehlo ]

module {
func.func @main(%arg0 : tensor<5xf32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>) -> tensor<5xf32> {
  %0 = "tf.ClipByValue"(%arg0, %arg1, %arg2) : (tensor<5xf32>, tensor<f32>, tensor<f32>) -> tensor<5xf32>
  func.return %0 : tensor<5xf32>
}
}