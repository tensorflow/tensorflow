// RUN: odml-to-stablehlo-opt %s -stablehlo-composite-legalize-tfl-custom | FileCheck %s
// RUN: tf_tfl_translate --enable-hlo-to-tf-conversion --input-mlir %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s --check-prefix=CHECK-ROUNDTRIP

module {
  func.func public @main(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>, %arg2: tensor<1x100x32x4xf32>,
      %arg3: tensor<1x500x4x4xf32>, %arg4: tensor<1x500x4x4xf32>, %arg5: tensor<1x1x100x500xf32>, %arg6: tensor<f32>)
      -> (tensor<3x3xf32>, tensor<1x100x32x4xf32>) {
    // CHECK-ROUNDTRIP: %0 = "tfl.custom"(%arg0, %arg1) {custom_code = "odml.update_kv_cache", custom_option = #tfl<const_bytes : "0x6B765F63616368655F6D617800010E00020001000100F40105032501">} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK-ROUNDTRIP: %1 = "tfl.custom"(%arg2, %arg3, %arg4, %arg5, %arg6) {custom_code = "odml.scaled_dot_product_attention", custom_option = #tfl<const_bytes : "0x00000100002401">} : (tensor<1x100x32x4xf32>, tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>, tensor<1x1x100x500xf32>, tensor<f32>) -> tensor<1x100x32x4xf32>
    %0 = func.call @test_kv_cache(%arg0, %arg1) : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    %1 = func.call @test_sdpa(%arg2, %arg3, %arg4, %arg5, %arg6) : (tensor<1x100x32x4xf32>,  tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>, tensor<1x1x100x500xf32>, tensor<f32>) -> tensor<1x100x32x4xf32>
    return %0, %1 : tensor<3x3xf32>, tensor<1x100x32x4xf32>
  }

  // CHECK-LABEL: func.func private @test_kv_cache
  func.func private @test_kv_cache(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
    // CHECK: %0 = "tfl.custom"(%arg0, %arg1) {custom_code = "odml.update_kv_cache", custom_option = #tfl<const_bytes : "0x6B765F63616368655F6D617800010E00020001000100F40105032501">} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    %0 = stablehlo.composite "odml.update_kv_cache" %arg0, %arg1 {composite_attributes = {kv_cache_max = 500 : i64}, decomposition = @odml.update_kv_cache.impl} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    return %0 : tensor<3x3xf32>
  }
  func.func private @odml.update_kv_cache.impl(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
    // No decomposition provided for test case.
    return %arg0 : tensor<3x3xf32>
  }

  // CHECK-LABEL: func.func private @test_sdpa
  func.func private @test_sdpa(%arg0: tensor<1x100x32x4xf32>, %arg1: tensor<1x500x4x4xf32>, %arg2: tensor<1x500x4x4xf32>, %arg3: tensor<1x1x100x500xf32>, %arg4: tensor<f32>) -> tensor<1x100x32x4xf32> {
    // CHECK:  %0 = "tfl.custom"(%arg0, %arg1, %arg2, %arg3, %arg4) {custom_code = "odml.scaled_dot_product_attention", custom_option = #tfl<const_bytes : "0x00000100002401">} : (tensor<1x100x32x4xf32>, tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>, tensor<1x1x100x500xf32>, tensor<f32>) -> tensor<1x100x32x4xf32>
    %0 = stablehlo.composite "odml.scaled_dot_product_attention" %arg0, %arg1, %arg2, %arg3, %arg4 {decomposition = @odml.scaled_dot_product_attention.impl} : (tensor<1x100x32x4xf32>, tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>, tensor<1x1x100x500xf32>, tensor<f32>) -> tensor<1x100x32x4xf32>
    return %0 : tensor<1x100x32x4xf32>
  }
  func.func private @odml.scaled_dot_product_attention.impl(%arg0: tensor<1x100x32x4xf32>, %arg1: tensor<1x500x4x4xf32>, %arg2: tensor<1x500x4x4xf32>, %arg3: tensor<1x1x100x500xf32>, %arg4: tensor<f32>) -> tensor<1x100x32x4xf32> {
    // No decomposition provided for test case.
    return %arg0 : tensor<1x100x32x4xf32>
  }

}