// RUN: xla-opt %s --split-input-file | FileCheck %s

tt.func @tma_descriptor_params(%arg0: tensor<512x128xf32>)
  -> tensor<512x128xf32> attributes {
    tma = #triton_xla.tma_descriptor<
      global_shape = [512, 128],
      block_shape = [32, 64],
      layout = [0, 1],
      element_byte_size = 4>
  } {
  tt.return %arg0  : tensor<512x128xf32>
}
// CHECK:  #tma_descriptor =  #triton_xla.tma_descriptor<
// CHECK-SAME:   global_shape = [512, 128],
// CHECK-SAME:   block_shape = [32, 64],
// CHECK-SAME:   layout = [0, 1],
// CHECK-SAME:   element_byte_size = 4>

// -----

tt.func @tma_descriptor_params(%arg0: tensor<512x128xf32>)
  -> tensor<512x128xf32> attributes {
    tma = #triton_xla.tma_descriptor<
      global_shape = [512, 128],
      block_shape = [32, 64],
      layout = [0, 1],
      element_byte_size = 4,
      swizzle_mode = "32b">
  } {
  tt.return %arg0  : tensor<512x128xf32>
}
// CHECK:  #tma_descriptor =  #triton_xla.tma_descriptor<
// CHECK-SAME:   global_shape = [512, 128],
// CHECK-SAME:   block_shape = [32, 64],
// CHECK-SAME:   layout = [0, 1],
// CHECK-SAME:   element_byte_size = 4,
// CHECK-SAME:   swizzle_mode = "32b">