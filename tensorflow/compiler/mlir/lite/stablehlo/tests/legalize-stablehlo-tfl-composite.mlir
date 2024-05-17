// RUN: odml-to-stablehlo-opt %s -stablehlo-composite-legalize-tfl-custom | FileCheck %s
// RUN: tf_tfl_translate --enable-hlo-to-tf-conversion --input-mlir %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s --check-prefix=CHECK-ROUNDTRIP

module {
  func.func public @main(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>, %arg2: tensor<1x100x32x4xf32>,
      %arg3: tensor<1x500x4x4xf32>, %arg4: tensor<1x500x4x4xf32>, %arg5: tensor<1x1x100x500xf32>, %arg6: tensor<f32>)
      -> tensor<1x100x32x4xf32> {
    // CHECK-ROUNDTRIP: %0 = "tfl.custom"(%arg2, %arg3, %arg4, %arg5, %arg6) <{custom_code = "odml.scaled_dot_product_attention", custom_option = #tfl<const_bytes : "0x00000100002401">}> : (tensor<1x100x32x4xf32>, tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>, tensor<1x1x100x500xf32>, tensor<f32>) -> tensor<1x100x32x4xf32>
    %0 = func.call @test_sdpa(%arg2, %arg3, %arg4, %arg5, %arg6) : (tensor<1x100x32x4xf32>,  tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>, tensor<1x1x100x500xf32>, tensor<f32>) -> tensor<1x100x32x4xf32>
    return %0: tensor<1x100x32x4xf32>
  }

  // CHECK-LABEL: func.func private @test_sdpa
  func.func private @test_sdpa(%arg0: tensor<1x100x32x4xf32>, %arg1: tensor<1x500x4x4xf32>, %arg2: tensor<1x500x4x4xf32>, %arg3: tensor<1x1x100x500xf32>, %arg4: tensor<f32>) -> tensor<1x100x32x4xf32> {
    // CHECK:  %0 = "tfl.custom"(%arg0, %arg1, %arg2, %arg3, %arg4) <{custom_code = "odml.scaled_dot_product_attention", custom_option = #tfl<const_bytes : "0x00000100002401">}> : (tensor<1x100x32x4xf32>, tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>, tensor<1x1x100x500xf32>, tensor<f32>) -> tensor<1x100x32x4xf32>
    %0 = stablehlo.composite "odml.scaled_dot_product_attention" %arg0, %arg1, %arg2, %arg3, %arg4 {decomposition = @odml.scaled_dot_product_attention.impl} : (tensor<1x100x32x4xf32>, tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>, tensor<1x1x100x500xf32>, tensor<f32>) -> tensor<1x100x32x4xf32>
    return %0 : tensor<1x100x32x4xf32>
  }
  func.func private @odml.scaled_dot_product_attention.impl(%arg0: tensor<1x100x32x4xf32>, %arg1: tensor<1x500x4x4xf32>, %arg2: tensor<1x500x4x4xf32>, %arg3: tensor<1x1x100x500xf32>, %arg4: tensor<f32>) -> tensor<1x100x32x4xf32> {
    // No decomposition provided for test case.
    return %arg0 : tensor<1x100x32x4xf32>
  }

  // CHECK-LABEL: func.func private @test_multiple_kv_caches
  func.func private @test_multiple_kv_caches(%arg0: tensor<1x500x4x4xf32>, %arg1: tensor<1x500x4x4xf32>, %arg2: tensor<100xi64>, %arg3: tensor<1x100x4x4xf32>, %arg4: tensor<1x100x4x4xf32>) -> (tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>) {
    // CHECK: %0:2 = "tfl.custom"(%arg2, %arg3, %arg4) <{custom_code = "odml.update_kv_cache", custom_option = #tfl<const_bytes : "0x6B765F63616368655F6D6178006C617965725F696E646578006E756D5F6C6179657273000325190E030001000300F40100000200050505092501">}> : (tensor<100xi64>, tensor<1x100x4x4xf32>, tensor<1x100x4x4xf32>) -> (tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>)
    // CHECK: %1:2 = "tfl.custom"(%arg2, %arg3, %arg4) <{custom_code = "odml.update_kv_cache", custom_option = #tfl<const_bytes : "0x6B765F63616368655F6D6178006C617965725F696E646578006E756D5F6C6179657273000325190E030001000300F40101000200050505092501">}> : (tensor<100xi64>, tensor<1x100x4x4xf32>, tensor<1x100x4x4xf32>) -> (tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>)
    %0:2 = stablehlo.composite "odml.update_kv_cache" %arg0, %arg1, %arg2, %arg3, %arg4 {composite_attributes = {kv_cache_max = 500 : i64}, decomposition = @odml.update_kv_cache.impl_0} : (tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>, tensor<100xi64>, tensor<1x100x4x4xf32>, tensor<1x100x4x4xf32>) -> (tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>)
    %1:2 = stablehlo.composite "odml.update_kv_cache" %0#0, %0#1, %arg2, %arg3, %arg4 {composite_attributes = {kv_cache_max = 500 : i64}, decomposition = @odml.update_kv_cache.impl_0} : (tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>, tensor<100xi64>, tensor<1x100x4x4xf32>, tensor<1x100x4x4xf32>) -> (tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>)
    return %1#0, %1#1 : tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>
  }
  func.func private @odml.update_kv_cache.impl_0(%arg0: tensor<1x500x4x4xf32>, %arg1: tensor<1x500x4x4xf32>, %arg2: tensor<100xi64>, %arg3: tensor<1x100x4x4xf32>, %arg4: tensor<1x100x4x4xf32>) -> (tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>) {
    %0 = stablehlo.constant dense<500> : tensor<100xi64>
    %1 = stablehlo.constant dense<0> : tensor<100xi64>
    %2 = stablehlo.compare  LT, %arg2, %1 : (tensor<100xi64>, tensor<100xi64>) -> tensor<100xi1>
    %3 = stablehlo.add %arg2, %0 : tensor<100xi64>
    %4 = stablehlo.select %2, %3, %arg2 : tensor<100xi1>, tensor<100xi64>
    %5 = stablehlo.reshape %4 : (tensor<100xi64>) -> tensor<100x1xi64>
    %6 = "stablehlo.scatter"(%arg0, %5, %arg3) ({
    ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
      stablehlo.return %arg6 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<1x500x4x4xf32>, tensor<100x1xi64>, tensor<1x100x4x4xf32>) -> tensor<1x500x4x4xf32>
    %7 = "stablehlo.scatter"(%arg1, %5, %arg4) ({
    ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
      stablehlo.return %arg6 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false} : (tensor<1x500x4x4xf32>, tensor<100x1xi64>, tensor<1x100x4x4xf32>) -> tensor<1x500x4x4xf32>
    return %6, %7 : tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>
  }

}