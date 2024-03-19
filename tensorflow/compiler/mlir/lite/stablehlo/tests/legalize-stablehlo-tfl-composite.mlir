// RUN: odml-to-stablehlo-opt %s -stablehlo-tfl | FileCheck %s

module {
  // CHECK-LABEL: test_kv_cache
  func.func @test_kv_cache(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
    // CHECK: %0 = "tfl.custom"(%arg0, %arg1) {custom_code = "odml.update_kv_cache", custom_option = #tfl<const_bytes : "0x6B765F63616368655F6D617800010E00020001000100F40105032501">} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    %0 = stablehlo.composite "odml.update_kv_cache" %arg0, %arg1 {composite_attributes = {kv_cache_max = 500 : i64}, decomposition = @odml.update_kv_cache.impl} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    return %0 : tensor<3x3xf32>
  }
  func.func @odml.update_kv_cache.impl(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
    // No decomposition provided for test case.
    return %arg0 : tensor<3x3xf32>
  }
  // CHECK-LABEL: test_sdpa
  func.func @test_sdpa(%arg0: tensor<1x100x32x4xf32>, %arg1: tensor<1x500x4x4xf32>, %arg2: tensor<1x500x4x4xf32>, %arg3: tensor<1x1x100x500xf32>, %arg4: tensor<f32>) -> tensor<1x100x32x4xf32> {
    // CHECK:  %0 = "tfl.custom"(%arg0, %arg1, %arg2, %arg3, %arg4) {custom_code = "odml.scaled_dot_product_attention", custom_option = #tfl<const_bytes : "0x00000100002401">} : (tensor<1x100x32x4xf32>, tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>, tensor<1x1x100x500xf32>, tensor<f32>) -> tensor<1x100x32x4xf32>
    %0 = stablehlo.composite "odml.scaled_dot_product_attention" %arg0, %arg1, %arg2, %arg3, %arg4 {decomposition = @odml.scaled_dot_product_attention.impl} : (tensor<1x100x32x4xf32>, tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>, tensor<1x1x100x500xf32>, tensor<f32>) -> tensor<1x100x32x4xf32>
    return %0 : tensor<1x100x32x4xf32>
  }
  func.func @odml.scaled_dot_product_attention.impl(%arg0: tensor<1x100x32x4xf32>, %arg1: tensor<1x500x4x4xf32>, %arg2: tensor<1x500x4x4xf32>, %arg3: tensor<1x1x100x500xf32>, %arg4: tensor<f32>) -> tensor<1x100x32x4xf32> {
    // No decomposition provided for test case.
    return %arg0 : tensor<1x100x32x4xf32>
  }
}

