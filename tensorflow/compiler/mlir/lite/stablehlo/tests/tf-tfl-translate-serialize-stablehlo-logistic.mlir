//RUN: tf_tfl_translate --enable-stablehlo-conversion --input-mlir %s -o /tmp/temp.stablehlo; [ -f /tmp/temp.stablehlo ]


module {
func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "tf.Sigmoid"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}
}
