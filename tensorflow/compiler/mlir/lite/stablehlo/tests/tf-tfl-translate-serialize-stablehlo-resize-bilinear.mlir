//RUN: tf_tfl_translate --enable-stablehlo-conversion --input-mlir %s -o /tmp/temp.stablehlo; [ -f /tmp/temp.stablehlo ]


module {
func.func @main(%arg0: tensor<21x32x32x128xf32>, %arg1: tensor<2xi32>) -> tensor<1x64x64x128xf32> {
  %0 = "tf.ResizeBilinear"(%arg0, %arg1) {align_corners = false, device = "", half_pixel_centers = true} : (tensor<21x32x32x128xf32>, tensor<2xi32>) -> tensor<1x64x64x128xf32>
  func.return %0 : tensor<1x64x64x128xf32>
}
}
