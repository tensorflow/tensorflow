module {
func.func @main(%arg0: tensor<1x128x4x256xf32>, %arg1: tensor<4xi32>, %arg2: tensor<4xi32>, %arg3: tensor<4xi32>) -> tensor<1x128x4x128xf32> {
  %0 = "tfl.strided_slice"(%arg0, %arg1, %arg2, %arg3) <{begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, offset = false, shrink_axis_mask = 0 : i32}> : (tensor<1x128x4x256xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x128x4x128xf32>
  return %0 : tensor<1x128x4x128xf32>
}
}