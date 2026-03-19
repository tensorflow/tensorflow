// RUN: odml-to-stablehlo-opt %s -stablehlo-composite-legalize-tfl-custom | FileCheck %s
module {
  // CHECK-LABEL: func.func private @test_quantize_and_dequantize
  func.func private @test_quantize_and_dequantize(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %0 = "tfl.pseudo_const"() <{value = dense<1.000000e+00> : tensor<4x3xf32>}> : () -> tensor<4x3xf32>
    // CHECK: %1:3 = "tfl.custom"(%0) <{custom_code = "odml.quantize_and_dequantize", custom_option = #tfl<const_bytes : "0x61786973006269747300020B0702010200040404042401">}> : (tensor<4x3xf32>) -> (tensor<4x3xf32>, tensor<4x3xf32>, tensor<1x3xf32>)
    %1:3 = stablehlo.composite "odml.quantize_and_dequantize" %0 {composite_attributes = {axis = 0 : i64, bits = 4 : i64}, decomposition = @call_module_odml.quantize_and_dequantize.0} : (tensor<4x3xf32>) -> (tensor<4x3xf32>, tensor<4x3xf32>, tensor<1x3xf32>)
    %2 = "tfl.batch_matmul"(%arg0, %1#0) <{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}> : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    return %2 : tensor<1x3xf32>
  }

  func.func private @call_module_odml.quantize_and_dequantize.0(%arg0: tensor<4x3xf32>) -> (tensor<4x3xf32>, tensor<4x3xf32>, tensor<1x3xf32>) {
    %0 = "tfl.abs"(%arg0) : (tensor<4x3xf32>) -> tensor<4x3xf32>
    %1 = "tfl.pseudo_const"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = "tfl.reduce_max"(%0, %1) <{keep_dims = false}> : (tensor<4x3xf32>, tensor<1xi32>) -> tensor<3xf32>
    %3 = "tfl.pseudo_const"() <{value = dense<0.142857149> : tensor<f32>}> : () -> tensor<f32>
    %4 = tfl.mul(%2, %3) <{fused_activation_function = "NONE"}> : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
    %5 = "tfl.pseudo_const"() <{value = dense<9.99999993E-9> : tensor<f32>}> : () -> tensor<f32>
    %6 = tfl.add(%4, %5) <{fused_activation_function = "NONE"}> : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
    %7 = "tfl.pseudo_const"() <{value = dense<[1, 3]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %8 = "tfl.reshape"(%6, %7) : (tensor<3xf32>, tensor<2xi32>) -> tensor<1x3xf32>
    %9 = tfl.div(%arg0, %8) <{fused_activation_function = "NONE"}> : (tensor<4x3xf32>, tensor<1x3xf32>) -> tensor<4x3xf32>
    %10 = tfl.sub %9, %9 {fused_activation_function = "NONE"} : tensor<4x3xf32>
    %11 = "tfl.round"(%9) : (tensor<4x3xf32>) -> tensor<4x3xf32>
    %12 = tfl.add %10, %11 {fused_activation_function = "NONE"} : tensor<4x3xf32>
    %13 = "tfl.pseudo_const"() <{value = dense<-8.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %14 = "tfl.maximum"(%12, %13) : (tensor<4x3xf32>, tensor<f32>) -> tensor<4x3xf32>
    %15 = "tfl.pseudo_const"() <{value = dense<7.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %16 = "tfl.minimum"(%14, %15) : (tensor<4x3xf32>, tensor<f32>) -> tensor<4x3xf32>
    %17 = tfl.mul(%16, %8) <{fused_activation_function = "NONE"}> : (tensor<4x3xf32>, tensor<1x3xf32>) -> tensor<4x3xf32>
    return %17, %16, %8 : tensor<4x3xf32>, tensor<4x3xf32>, tensor<1x3xf32>
  }
}
