module {
func.func @main(%arg0: tensor<1x54x72x96xf32>) -> tensor<1x108x144x96xf32> {
  %cst = "tfl.pseudo_const"() <{value = dense<[108, 144]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %0 = "tfl.resize_nearest_neighbor"(%arg0, %cst) <{align_corners = false, half_pixel_centers = true}> : (tensor<1x54x72x96xf32>, tensor<2xi32>) -> tensor<1x108x144x96xf32>
  return %0 : tensor<1x108x144x96xf32>
}
}