// RUN: odml-to-stablehlo-opt %s -stablehlo-composite-legalize-tfl-custom | FileCheck %s

func.func private @odml.update_kv_cache.impl_0(%arg0: tensor<1x500x4x4xf32>, %arg1: tensor<1x500x4x4xf32>, %arg2: tensor<100xi64>, %arg3: tensor<1x100x4x4xf32>, %arg4: tensor<1x100x4x4xf32>) -> (tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>)
// CHECK-LABEL: func.func private @test_multiple_kv_caches
func.func private @test_multiple_kv_caches(%arg0: tensor<1x500x4x4xf32>, %arg1: tensor<1x500x4x4xf32>, %arg2: tensor<100xi64>, %arg3: tensor<1x100x4x4xf32>, %arg4: tensor<1x100x4x4xf32>) -> (tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>) {
  // CHECK: %0:2 = "tfl.custom"(%arg2, %arg3, %arg4) <{custom_code = "odml.update_kv_cache", custom_option = #tfl<const_bytes : "0x6B765F63616368655F6D6178006C617965725F696E646578006E756D5F6C6179657273000325190E030001000300F40100000200050505092501">}> : (tensor<100xi64>, tensor<1x100x4x4xf32>, tensor<1x100x4x4xf32>) -> (tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>)
  // CHECK: %1:2 = "tfl.custom"(%arg2, %arg3, %arg4) <{custom_code = "odml.update_kv_cache", custom_option = #tfl<const_bytes : "0x6B765F63616368655F6D6178006C617965725F696E646578006E756D5F6C6179657273000325190E030001000300F40101000200050505092501">}> : (tensor<100xi64>, tensor<1x100x4x4xf32>, tensor<1x100x4x4xf32>) -> (tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>)
  %0:2 = stablehlo.composite "odml.update_kv_cache" %arg0, %arg1, %arg2, %arg3, %arg4 {composite_attributes = {kv_cache_max = 500 : i64}, decomposition = @odml.update_kv_cache.impl_0} : (tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>, tensor<100xi64>, tensor<1x100x4x4xf32>, tensor<1x100x4x4xf32>) -> (tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>)
  %1:2 = stablehlo.composite "odml.update_kv_cache" %0#0, %0#1, %arg2, %arg3, %arg4 {composite_attributes = {kv_cache_max = 500 : i64}, decomposition = @odml.update_kv_cache.impl_0} : (tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>, tensor<100xi64>, tensor<1x100x4x4xf32>, tensor<1x100x4x4xf32>) -> (tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>)
  return %1#0, %1#1 : tensor<1x500x4x4xf32>, tensor<1x500x4x4xf32>
}

// ---

func.func private @test_odml_detector.detector.impl_0(%arg0: tensor<2xf32>) -> tensor<2xf32>
// CHECK-LABEL: func.func private @test_odml_detector
func.func @test_odml_detector(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>) {
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<2xf32>
  // CHECK %1 = "tfl.custom"(%0) <{custom_code = "odml.detector", custom_option = #tfl<const_bytes : "0x6E616D6500036F757400776F726B696E675F64697200082F746D702F7473740002211802010220101414042401">}> : (tensor<2xf32>) -> tensor<2xf32>
  %1 = stablehlo.composite "odml.detector" %0 {composite_attributes = {name = "out", working_dir = "/tmp/tst"}, decomposition = @test_odml_detector.detector.impl_0} : (tensor<2xf32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}