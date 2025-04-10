// RUN: tf-opt %s -split-input-file -tfl-prepare-quantize="qdq-conversion-mode=Strict" | FileCheck --check-prefix=QDQ-STRICT %s

// -----

// QDQ-STRICT-LABEL: func @test_bais_quantparams_propagation_negative
func.func @test_bais_quantparams_propagation_negative(%arg0: tensor<1x32x32x640xf32>, %arg1: tensor<1x32x32x640xf32>, %arg2: tensor<1x32x32x640xf32>, %arg3: tensor<1x32x32x640xf32>, %arg4: tensor<1x32x32x1280xf32>) -> (tensor<1x32x32x640xf32>, tensor<1x32x32x1920xf32>) {
  %cst_0 = arith.constant dense<[1, 5, 5, 640]> : tensor<4xi32>
  %cst_1 = arith.constant dense<0.0010000000474974513> : tensor<5x5x1x640xf32>
  %cst_2 = arith.constant dense<0.0> : tensor<640xf32>
  %0 = tfl.add %arg2, %arg3 {fused_activation_function = "NONE"} : tensor<1x32x32x640xf32>
  %1 = "tfl.quantize"(%cst_1) <{qtype = tensor<5x5x1x640x!quant.uniform<i4:f32, 0.0010000000474974513>>}> : (tensor<5x5x1x640xf32>) -> tensor<5x5x1x640x!quant.uniform<i4:f32, 0.0010000000474974513>>
  %2 = "tfl.dequantize"(%1) : (tensor<5x5x1x640x!quant.uniform<i4:f32, 0.0010000000474974513>>) -> tensor<5x5x1x640xf32>
  %3 = "tfl.reshape"(%2, %cst_0) : (tensor<5x5x1x640xf32>, tensor<4xi32>) -> tensor<1x5x5x640xf32>
  %4 = "tfl.depthwise_conv_2d"(%0, %3, %cst_2) <{depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x32x32x640xf32>, tensor<1x5x5x640xf32>, tensor<640xf32>) -> tensor<1x32x32x640xf32>
  %5 = tfl.mul %4, %4 {fused_activation_function = "NONE"} : tensor<1x32x32x640xf32>
  %6 = "tfl.concatenation"(%0, %arg4) <{axis = 3 : i32, fused_activation_function = "NONE"}> : (tensor<1x32x32x640xf32>, tensor<1x32x32x1280xf32>) -> tensor<1x32x32x1920xf32>
  %7 = "tfl.quantize"(%6) <{qtype = tensor<1x32x32x1920x!quant.uniform<i8:f32, 6.7734990119934082>>}> : (tensor<1x32x32x1920xf32>) -> tensor<1x32x32x1920x!quant.uniform<i8:f32, 6.7734990119934082>>
  %8 = "tfl.dequantize"(%7) : (tensor<1x32x32x1920x!quant.uniform<i8:f32, 6.7734990119934082>>) -> tensor<1x32x32x1920xf32>
  return %5, %8 : tensor<1x32x32x640xf32>, tensor<1x32x32x1920xf32>
  // QDQ-STRICT-CHECK: %0 = "tfl.quantize"(%arg4) <{qtype = tensor<1x32x32x1280x!quant.uniform<i8:f32, 6.7734990119934082>>}> {volatile} : (tensor<1x32x32x1280xf32>) -> tensor<1x32x32x1280x!quant.uniform<i8:f32, 6.7734990119934082>>
  // QDQ-STRICT-CHECK: %1 = "tfl.dequantize"(%0) : (tensor<1x32x32x1280x!quant.uniform<i8:f32, 6.7734990119934082>>) -> tensor<1x32x32x1280xf32>
  // QDQ-STRICT-CHECK: %cst = arith.constant dense<[1, 5, 5, 640]> : tensor<4xi32>
  // QDQ-STRICT-CHECK: %cst_0 = arith.constant dense<1.000000e-03> : tensor<5x5x1x640xf32>
  // QDQ-STRICT-CHECK: %cst_1 = arith.constant dense<0.000000e+00> : tensor<640xf32>
  // QDQ-STRICT-CHECK: %2 = tfl.add %arg2, %arg3 {fused_activation_function = "NONE"} : tensor<1x32x32x640xf32>
  // QDQ-STRICT-CHECK: %3 = "tfl.quantize"(%2) <{qtype = tensor<1x32x32x640x!quant.uniform<i8:f32, 6.7734990119934082>>}> {volatile} : (tensor<1x32x32x640xf32>) -> tensor<1x32x32x640x!quant.uniform<i8:f32, 6.7734990119934082>>
  // QDQ-STRICT-CHECK: %4 = "tfl.dequantize"(%3) : (tensor<1x32x32x640x!quant.uniform<i8:f32, 6.7734990119934082>>) -> tensor<1x32x32x640xf32>
  // QDQ-STRICT-CHECK: %5 = "tfl.quantize"(%cst_0) <{qtype = tensor<5x5x1x640x!quant.uniform<i4:f32, 0.0010000000474974513>>}> : (tensor<5x5x1x640xf32>) -> tensor<5x5x1x640x!quant.uniform<i4:f32, 0.0010000000474974513>>
  // QDQ-STRICT-CHECK: %6 = "tfl.dequantize"(%5) : (tensor<5x5x1x640x!quant.uniform<i4:f32, 0.0010000000474974513>>) -> tensor<5x5x1x640xf32>
  // QDQ-STRICT-CHECK: %7 = "tfl.reshape"(%6, %cst) : (tensor<5x5x1x640xf32>, tensor<4xi32>) -> tensor<1x5x5x640xf32>
  // QDQ-STRICT-CHECK: %8 = "tfl.quantize"(%7) <{qtype = tensor<1x5x5x640x!quant.uniform<i4:f32, 0.0010000000474974513>>}> {volatile} : (tensor<1x5x5x640xf32>) -> tensor<1x5x5x640x!quant.uniform<i4:f32, 0.0010000000474974513>>
  // QDQ-STRICT-CHECK: %9 = "tfl.dequantize"(%8) : (tensor<1x5x5x640x!quant.uniform<i4:f32, 0.0010000000474974513>>) -> tensor<1x5x5x640xf32>
  // QDQ-STRICT-CHECK: %10 = "tfl.depthwise_conv_2d"(%4, %9, %cst_1) <{depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x32x32x640xf32>, tensor<1x5x5x640xf32>, tensor<640xf32>) -> tensor<1x32x32x640xf32>
  // QDQ-STRICT-CHECK: %11 = tfl.mul %10, %10 {fused_activation_function = "NONE"} : tensor<1x32x32x640xf32>
  // QDQ-STRICT-CHECK: %12 = "tfl.concatenation"(%4, %1) <{axis = 3 : i32, fused_activation_function = "NONE"}> : (tensor<1x32x32x640xf32>, tensor<1x32x32x1280xf32>) -> tensor<1x32x32x1920xf32>
  // QDQ-STRICT-CHECK: %13 = "tfl.quantize"(%12) <{qtype = tensor<1x32x32x1920x!quant.uniform<i8:f32, 6.7734990119934082>>}> : (tensor<1x32x32x1920xf32>) -> tensor<1x32x32x1920x!quant.uniform<i8:f32, 6.7734990119934082>>
  // QDQ-STRICT-CHECK: %14 = "tfl.dequantize"(%13) : (tensor<1x32x32x1920x!quant.uniform<i8:f32, 6.7734990119934082>>) -> tensor<1x32x32x1920xf32>
  // QDQ-STRICT-CHECK: return %11, %14 : tensor<1x32x32x640xf32>, tensor<1x32x32x1920xf32>
}

// -----

// QDQ-STRICT-LABEL: func @test_bias_quantparams_propagation_positive
func.func @test_bias_quantparams_propagation_positive(%arg0: tensor<1x32x32x640xf32>, %arg1: tensor<1x32x32x640xf32>, %arg2: tensor<1x32x32x640xf32>, %arg3: tensor<1x32x32x640xf32>, %arg4: tensor<1x32x32x1280xf32>) -> (tensor<1x32x32x640xf32>, tensor<1x32x32x1920xf32>) {
  %cst_0 = arith.constant dense<[1, 5, 5, 640]> : tensor<4xi32>
  %cst_1 = arith.constant dense<0.0010000000474974513> : tensor<5x5x1x640xf32>
  %cst_2 = arith.constant dense<3.50> : tensor<640xf32>
  %01 = "tfl.quantize"(%cst_2) <{qtype = tensor<640x!quant.uniform<i8:f32, 0.001000000065758793>>}> : (tensor<640xf32>) -> tensor<640x!quant.uniform<i8:f32, 0.001000000065758793>>
  %02 = "tfl.dequantize"(%01) : (tensor<640x!quant.uniform<i8:f32, 0.001000000065758793>>) -> tensor<640xf32>
  %0 = tfl.add %arg2, %arg3 {fused_activation_function = "NONE"} : tensor<1x32x32x640xf32>
  %1 = "tfl.quantize"(%cst_1) <{qtype = tensor<5x5x1x640x!quant.uniform<i4:f32, 0.0010000000474974513>>}> : (tensor<5x5x1x640xf32>) -> tensor<5x5x1x640x!quant.uniform<i4:f32, 0.0010000000474974513>>
  %2 = "tfl.dequantize"(%1) : (tensor<5x5x1x640x!quant.uniform<i4:f32, 0.0010000000474974513>>) -> tensor<5x5x1x640xf32>
  %3 = "tfl.reshape"(%2, %cst_0) : (tensor<5x5x1x640xf32>, tensor<4xi32>) -> tensor<1x5x5x640xf32>
  %4 = "tfl.depthwise_conv_2d"(%0, %3, %02) <{depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x32x32x640xf32>, tensor<1x5x5x640xf32>, tensor<640xf32>) -> tensor<1x32x32x640xf32>
  %5 = tfl.mul %4, %4 {fused_activation_function = "NONE"} : tensor<1x32x32x640xf32>
  %6 = "tfl.concatenation"(%0, %arg4) <{axis = 3 : i32, fused_activation_function = "NONE"}> : (tensor<1x32x32x640xf32>, tensor<1x32x32x1280xf32>) -> tensor<1x32x32x1920xf32>
  %7 = "tfl.quantize"(%6) <{qtype = tensor<1x32x32x1920x!quant.uniform<i8:f32, 6.7734990119934082>>}> : (tensor<1x32x32x1920xf32>) -> tensor<1x32x32x1920x!quant.uniform<i8:f32, 6.7734990119934082>>
  %8 = "tfl.dequantize"(%7) : (tensor<1x32x32x1920x!quant.uniform<i8:f32, 6.7734990119934082>>) -> tensor<1x32x32x1920xf32>
  return %5, %8 : tensor<1x32x32x640xf32>, tensor<1x32x32x1920xf32>
  // QDQ-STRICT-CHECK: %0 = "tfl.quantize"(%arg4) <{qtype = tensor<1x32x32x1280x!quant.uniform<i8:f32, 6.7734990119934082>>}> {volatile} : (tensor<1x32x32x1280xf32>) -> tensor<1x32x32x1280x!quant.uniform<i8:f32, 6.7734990119934082>>
  // QDQ-STRICT-CHECK: %1 = "tfl.dequantize"(%0) : (tensor<1x32x32x1280x!quant.uniform<i8:f32, 6.7734990119934082>>) -> tensor<1x32x32x1280xf32>
  // QDQ-STRICT-CHECK: %cst = arith.constant dense<[1, 5, 5, 640]> : tensor<4xi32>
  // QDQ-STRICT-CHECK: %cst_0 = arith.constant dense<1.000000e-03> : tensor<5x5x1x640xf32>
  // QDQ-STRICT-CHECK: %cst_1 = arith.constant dense<3.500000e+00> : tensor<640xf32>
  // QDQ-STRICT-CHECK: %2 = "tfl.quantize"(%cst_1) <{qtype = tensor<640x!quant.uniform<i8:f32, 0.001000000065758793>>}> : (tensor<640xf32>) -> tensor<640x!quant.uniform<i8:f32, 0.001000000065758793>>
  // QDQ-STRICT-CHECK: %3 = "tfl.dequantize"(%2) : (tensor<640x!quant.uniform<i8:f32, 0.001000000065758793>>) -> tensor<640xf32>
  // QDQ-STRICT-CHECK: %4 = tfl.add %arg2, %arg3 {fused_activation_function = "NONE"} : tensor<1x32x32x640xf32>
  // QDQ-STRICT-CHECK: %5 = "tfl.quantize"(%4) <{qtype = tensor<1x32x32x640x!quant.uniform<i8:f32, 6.7734990119934082>>}> {volatile} : (tensor<1x32x32x640xf32>) -> tensor<1x32x32x640x!quant.uniform<i8:f32, 6.7734990119934082>>
  // QDQ-STRICT-CHECK: %6 = "tfl.dequantize"(%5) : (tensor<1x32x32x640x!quant.uniform<i8:f32, 6.7734990119934082>>) -> tensor<1x32x32x640xf32>
  // QDQ-STRICT-CHECK: %7 = "tfl.quantize"(%cst_0) <{qtype = tensor<5x5x1x640x!quant.uniform<i4:f32, 0.0010000000474974513>>}> : (tensor<5x5x1x640xf32>) -> tensor<5x5x1x640x!quant.uniform<i4:f32, 0.0010000000474974513>>
  // QDQ-STRICT-CHECK: %8 = "tfl.dequantize"(%7) : (tensor<5x5x1x640x!quant.uniform<i4:f32, 0.0010000000474974513>>) -> tensor<5x5x1x640xf32>
  // QDQ-STRICT-CHECK: %9 = "tfl.reshape"(%8, %cst) : (tensor<5x5x1x640xf32>, tensor<4xi32>) -> tensor<1x5x5x640xf32>
  // QDQ-STRICT-CHECK: %10 = "tfl.quantize"(%9) <{qtype = tensor<1x5x5x640x!quant.uniform<i4:f32, 0.0010000000474974513>>}> {volatile} : (tensor<1x5x5x640xf32>) -> tensor<1x5x5x640x!quant.uniform<i4:f32, 0.0010000000474974513>>
  // QDQ-STRICT-CHECK: %11 = "tfl.dequantize"(%10) : (tensor<1x5x5x640x!quant.uniform<i4:f32, 0.0010000000474974513>>) -> tensor<1x5x5x640xf32>
  // QDQ-STRICT-CHECK: %12 = "tfl.depthwise_conv_2d"(%6, %11, %3) <{depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x32x32x640xf32>, tensor<1x5x5x640xf32>, tensor<640xf32>) -> tensor<1x32x32x640xf32>
  // QDQ-STRICT-CHECK: %13 = tfl.mul %12, %12 {fused_activation_function = "NONE"} : tensor<1x32x32x640xf32>
  // QDQ-STRICT-CHECK: %14 = "tfl.concatenation"(%6, %1) <{axis = 3 : i32, fused_activation_function = "NONE"}> : (tensor<1x32x32x640xf32>, tensor<1x32x32x1280xf32>) -> tensor<1x32x32x1920xf32>
  // QDQ-STRICT-CHECK: %15 = "tfl.quantize"(%14) <{qtype = tensor<1x32x32x1920x!quant.uniform<i8:f32, 6.7734990119934082>>}> : (tensor<1x32x32x1920xf32>) -> tensor<1x32x32x1920x!quant.uniform<i8:f32, 6.7734990119934082>>
  // QDQ-STRICT-CHECK: %16 = "tfl.dequantize"(%15) : (tensor<1x32x32x1920x!quant.uniform<i8:f32, 6.7734990119934082>>) -> tensor<1x32x32x1920xf32>
  // QDQ-STRICT-CHECK: return %13, %16 : tensor<1x32x32x640xf32>, tensor<1x32x32x1920xf32>
}
