// RUN: litert-opt %s -tfl-fuse-qdq | FileCheck %s
// CHECK-LABEL: QuantizeConvDRQ
func.func private @XlaCallModule_quant.fake_quant.impl_0(%arg0: tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
func.func @QuantizeConvDRQ(%arg0: tensor<1x4x4x3xf32>) -> (tensor<1x4x4x1xf32>) {
  // CHECK:    %cst = arith.constant dense<0.000000e+00> : tensor<1xf32>
  %cst = arith.constant dense<0.000000e+00> : tensor<1xf32>
  %cst_0 = arith.constant dense<[[[[1.76285899, -0.257785767, 0.20429258], [1.16310906, 0.23124367, 0.529797196]], [[0.348971426, -0.319283515, -0.772461354], [0.316666812, 1.88180697, -1.78054631]]]]> : tensor<1x2x2x3xf32>
  %0 = stablehlo.composite "quant.fake_quant" %arg0 {composite_attributes = {dtype = "i8", narrow_range = false, quantization_dimension = 0 : i32, scale = dense<> : tensor<0xf64>, zero_point = dense<> : tensor<0xi64>}, decomposition = @XlaCallModule_quant.fake_quant.impl_0} : (tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
  // CHECK{LITERAL}:    %0 = "tfl.pseudo_qconst"() <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>, value = dense<[[[[119, -17, 14], [78, 16, 36]], [[24, -22, -52], [21, 127, -120]]]]> : tensor<1x2x2x3xi8>}> : () -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
  %1 = "tfl.quantize"(%cst_0) <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>}> : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
  %2 = "tfl.dequantize"(%1) : (tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>) -> tensor<1x2x2x3xf32>
  // CHECK:    %1 = "tfl.conv_2d"(%arg0, %0, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x4x4x3xf32>, tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  %3 = "tfl.conv_2d"(%0, %2, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x4x4x3xf32>, tensor<1x2x2x3xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  // CHECK:    return %1 : tensor<1x4x4x1xf32>
  return %3 : tensor<1x4x4x1xf32>
}

// -----

// CHECK-LABEL: QuantizeConvDrqWithPad
func.func private @XlaCallModule_quant.fake_quant.impl_1(%arg0: tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
func.func @QuantizeConvDrqWithPad(%arg0: tensor<1x4x4x3xf32>) -> (tensor<1x6x6x1xf32>) {
  // CHECK:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xf32>
  %cst = arith.constant dense<0.000000e+00> : tensor<1xf32>
  %cst_0 = arith.constant dense<[[[[1.76285899, -0.257785767, 0.20429258], [1.16310906, 0.23124367, 0.529797196]], [[0.348971426, -0.319283515, -0.772461354], [0.316666812, 1.88180697, -1.78054631]]]]> : tensor<1x2x2x3xf32>
  %0 = stablehlo.composite "quant.fake_quant" %arg0 {composite_attributes = {dtype = "i8", narrow_range = false, quantization_dimension = 0 : i32, scale = dense<> : tensor<0xf64>, zero_point = dense<> : tensor<0xi64>}, decomposition = @XlaCallModule_quant.fake_quant.impl_1} : (tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
  // CHECK-LITERAL:    %cst = arith.constant dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>
  %paddings = arith.constant dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>
  // CHECK:    %0 = "tfl.pad"(%arg0, %cst) : (tensor<1x4x4x3xf32>, tensor<4x2xi32>) -> tensor<1x6x6x3xf32>
  %1 = "tfl.pad"(%0, %paddings) : (tensor<1x4x4x3xf32>, tensor<4x2xi32>) -> tensor<1x6x6x3xf32>
  // CHECK{LITERAL}:    %1 = "tfl.pseudo_qconst"() <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>, value = dense<[[[[119, -17, 14], [78, 16, 36]], [[24, -22, -52], [21, 127, -120]]]]> : tensor<1x2x2x3xi8>}> : () -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
  %2 = "tfl.quantize"(%cst_0) <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>}> : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
  %3 = "tfl.dequantize"(%2) : (tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>) -> tensor<1x2x2x3xf32>
  // CHECK:    %2 = "tfl.conv_2d"(%0, %1, %cst_0) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x6x6x3xf32>, tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>, tensor<1xf32>) -> tensor<1x6x6x1xf32>
  %4 = "tfl.conv_2d"(%1, %3, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x6x6x3xf32>, tensor<1x2x2x3xf32>, tensor<1xf32>) -> tensor<1x6x6x1xf32>
  // CHECK:    return %2 : tensor<1x6x6x1xf32>
  return %4 : tensor<1x6x6x1xf32>
}

// -----

// CHECK-LABEL: QuantizeConvWithBiasDRQ
func.func @QuantizeConvWithBiasDRQ(%arg0: tensor<1x4x4x3xf32>) -> (tensor<1x4x4x1xf32>) {
  // CHECK:    %cst = arith.constant dense<1.14751196> : tensor<1xf32>
  %cst = arith.constant dense<1.14751196> : tensor<1xf32>
  %cst_0 = arith.constant dense<[[[[1.76285899, -0.257785767, 0.20429258], [1.16310906, 0.23124367, 0.529797196]], [[0.348971426, -0.319283515, -0.772461354], [0.316666812, 1.88180697, -1.78054631]]]]> : tensor<1x2x2x3xf32>
  %0 = stablehlo.composite "quant.fake_quant" %arg0 {composite_attributes = {dtype = "i8", narrow_range = false, quantization_dimension = 0 : i32, scale = dense<> : tensor<0xf64>, zero_point = dense<> : tensor<0xi64>}, decomposition = @XlaCallModule_quant.fake_quant.impl_0} : (tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
  // CHECK{LITERAL}:    %0 = "tfl.pseudo_qconst"() <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>, value = dense<[[[[119, -17, 14], [78, 16, 36]], [[24, -22, -52], [21, 127, -120]]]]> : tensor<1x2x2x3xi8>}> : () -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
  %1 = "tfl.quantize"(%cst_0) <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>}> : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
  %2 = "tfl.dequantize"(%1) : (tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>) -> tensor<1x2x2x3xf32>
  // CHECK:    %1 = "tfl.conv_2d"(%arg0, %0, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x4x4x3xf32>, tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  %3 = "tfl.conv_2d"(%0, %2, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x4x4x3xf32>, tensor<1x2x2x3xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  // CHECK:    return %1 : tensor<1x4x4x1xf32>
  return %3 : tensor<1x4x4x1xf32>
}

// -----

// CHECK-LABEL: QuantizeConvWithBiasAndReluDRQ
func.func @QuantizeConvWithBiasAndReluDRQ(%arg0: tensor<1x4x4x3xf32>) -> (tensor<1x4x4x1xf32>) {
  // CHECK: %cst = arith.constant dense<1.14751196> : tensor<1xf32>
  %cst = arith.constant dense<1.14751196> : tensor<1xf32>
  %cst_0 = arith.constant dense<[[[[1.76285899, -0.257785767, 0.20429258], [1.16310906, 0.23124367, 0.529797196]], [[0.348971426, -0.319283515, -0.772461354], [0.316666812, 1.88180697, -1.78054631]]]]> : tensor<1x2x2x3xf32>
  %0 = stablehlo.composite "quant.fake_quant" %arg0 {composite_attributes = {dtype = "i8", narrow_range = false, quantization_dimension = 0 : i32, scale = dense<> : tensor<0xf64>, zero_point = dense<> : tensor<0xi64>}, decomposition = @XlaCallModule_quant.fake_quant.impl_0} : (tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
  // CHECK{LITERAL}: %0 = "tfl.pseudo_qconst"() <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>, value = dense<[[[[119, -17, 14], [78, 16, 36]], [[24, -22, -52], [21, 127, -120]]]]> : tensor<1x2x2x3xi8>}> : () -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
  %1 = "tfl.quantize"(%cst_0) <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>}> : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
  %2 = "tfl.dequantize"(%1) : (tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>) -> tensor<1x2x2x3xf32>
  // CHECK: %1 = "tfl.conv_2d"(%arg0, %0, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x4x4x3xf32>, tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  %3 = "tfl.conv_2d"(%0, %2, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x4x4x3xf32>, tensor<1x2x2x3xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  // CHECK: return %1 : tensor<1x4x4x1xf32>
  return %3 : tensor<1x4x4x1xf32>
}

// -----

// CHECK-LABEL: QuantizeConvWithBiasAndReluWeightOnly
func.func @QuantizeConvWithBiasAndReluWeightOnly(%arg0: tensor<1x4x4x3xf32>) -> (tensor<1x4x4x1xf32>) {
  // CHECK:  %cst = arith.constant dense<1.14751196> : tensor<1xf32>
  %cst = arith.constant dense<1.14751196> : tensor<1xf32>
  %cst_0 = arith.constant dense<[[[[1.76285899, -0.257785767, 0.20429258], [1.16310906, 0.23124367, 0.529797196]], [[0.348971426, -0.319283515, -0.772461354], [0.316666812, 1.88180697, -1.78054631]]]]> : tensor<1x2x2x3xf32>
  // CHECK{LITERAL}:  %0 = "tfl.pseudo_qconst"() <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>, value = dense<[[[[119, -17, 14], [78, 16, 36]], [[24, -22, -52], [21, 127, -120]]]]> : tensor<1x2x2x3xi8>}> : () -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
  %0 = "tfl.quantize"(%cst_0) <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>}> : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
  // CHECK:  %1 = "tfl.dequantize"(%0) : (tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>) -> tensor<1x2x2x3xf32>
  %1 = "tfl.dequantize"(%0) : (tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>) -> tensor<1x2x2x3xf32>
  // CHECK:  %2 = "tfl.conv_2d"(%arg0, %1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x4x4x3xf32>, tensor<1x2x2x3xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  %2 = "tfl.conv_2d"(%arg0, %1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x4x4x3xf32>, tensor<1x2x2x3xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  // CHECK:  return %2 : tensor<1x4x4x1xf32>
  return %2 : tensor<1x4x4x1xf32>
}

// -----

// CHECK-LABEL: QuantizeConvWithBiasAndReluSRQ
func.func @QuantizeConvWithBiasAndReluSRQ(%arg0: tensor<1x4x4x3xf32>) -> (tensor<1x4x4x1xf32>) {
  %cst = arith.constant dense<1.14751196> : tensor<1xf32>
  // CHECK: %0 = "tfl.pseudo_qconst"() <{qtype = tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>, value = dense<20578> : tensor<1xi32>}> : () -> tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>
  %0 = "tfl.quantize"(%cst) <{qtype = tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>}> : (tensor<1xf32>) -> tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>
  %1 = "tfl.dequantize"(%0) : (tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>) -> tensor<1xf32>
  %cst_0 = arith.constant dense<[[[[1.76285899, -0.257785767, 0.20429258], [1.16310906, 0.23124367, 0.529797196]], [[0.348971426, -0.319283515, -0.772461354], [0.316666812, 1.88180697, -1.78054631]]]]> : tensor<1x2x2x3xf32>
  // CHECK: %1 = "tfl.quantize"(%arg0) <{qtype = tensor<1x4x4x3x!quant.uniform<i8:f32, 0.0037634586915373802:-128>>}> : (tensor<1x4x4x3xf32>) -> tensor<1x4x4x3x!quant.uniform<i8:f32, 0.0037634586915373802:-128>>
  %2 = "tfl.quantize"(%arg0) <{qtype = tensor<1x4x4x3x!quant.uniform<i8:f32, 0.0037634586915373802:-128>>}> : (tensor<1x4x4x3xf32>) -> tensor<1x4x4x3x!quant.uniform<i8:f32, 0.0037634586915373802:-128>>
  %3 = "tfl.dequantize"(%2) : (tensor<1x4x4x3x!quant.uniform<i8:f32, 0.0037634586915373802:-128>>) -> tensor<1x4x4x3xf32>
  // CHECK{LITERAL}: %2 = "tfl.pseudo_qconst"() <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>, value = dense<[[[[119, -17, 14], [78, 16, 36]], [[24, -22, -52], [21, 127, -120]]]]> : tensor<1x2x2x3xi8>}> : () -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
  %4 = "tfl.quantize"(%cst_0) <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>}> : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
  %5 = "tfl.dequantize"(%4) : (tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>) -> tensor<1x2x2x3xf32>
  // CHECK: %3 = "tfl.conv_2d"(%1, %2, %0) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x4x4x3x!quant.uniform<i8:f32, 0.0037634586915373802:-128>>, tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>, tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>) -> tensor<1x4x4x1x!quant.uniform<i8:f32, 0.013401651754975319:-128>>
  %6 = "tfl.conv_2d"(%3, %5, %1) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x4x4x3xf32>, tensor<1x2x2x3xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  %7 = "tfl.quantize"(%6) <{qtype = tensor<1x4x4x1x!quant.uniform<i8:f32, 0.013401651754975319:-128>>}> : (tensor<1x4x4x1xf32>) -> tensor<1x4x4x1x!quant.uniform<i8:f32, 0.013401651754975319:-128>>
  // CHECK: %4 = "tfl.dequantize"(%3) : (tensor<1x4x4x1x!quant.uniform<i8:f32, 0.013401651754975319:-128>>) -> tensor<1x4x4x1xf32>
  %8 = "tfl.dequantize"(%7) : (tensor<1x4x4x1x!quant.uniform<i8:f32, 0.013401651754975319:-128>>) -> tensor<1x4x4x1xf32>
  // CHECK: return %4 : tensor<1x4x4x1xf32>
  return %8 : tensor<1x4x4x1xf32>
}

// -----

// CHECK-LABEL: QuantizeEmbeddingLookupDrq
func.func @QuantizeEmbeddingLookupDrq(%arg0: tensor<2xi32>) -> (tensor<2x4xf32>){
  %cst = arith.constant dense<[[1.0545162, -0.969288647, -0.594602108, -0.0318857245], [2.41093326, -1.87844908, -0.784769594, -0.313708425], [0.333708912, 1.76770353, -1.02776456, 1.41117179], [-0.508497119, -0.526377499, 0.503150403, 1.05497932], [-0.0874073281, 0.795816719, 2.65656161, -0.58229059]]> : tensor<5x4xf32>
  // CHECK{LITERAL}: %0 = "tfl.pseudo_qconst"() <{qtype = tensor<5x4x!quant.uniform<i8:f32:0, {0.0082384077832102776,0.018835416063666344,0.013810183852910995,0.0082420259714126587,0.020754387602210045}>>, value = dense<[[127, -118, -72, -4], [127, -100, -42, -17], [24, 127, -74, 102], [-62, -64, 61, 127], [-4, 38, 127, -28]]> : tensor<5x4xi8>}> : () -> tensor<5x4x!quant.uniform<i8:f32:0, {0.0082384077832102776,0.018835416063666344,0.013810183852910995,0.0082420259714126587,0.020754387602210045}>>
  %0 = "tfl.quantize"(%cst) <{qtype = tensor<5x4x!quant.uniform<i8:f32:0, {0.0082384077832102776,0.018835416063666344,0.013810183852910995,0.0082420259714126587,0.020754387602210045}>>}> : (tensor<5x4xf32>) -> tensor<5x4x!quant.uniform<i8:f32:0, {0.0082384077832102776,0.018835416063666344,0.013810183852910995,0.0082420259714126587,0.020754387602210045}>>
  %1 = "tfl.dequantize"(%0) : (tensor<5x4x!quant.uniform<i8:f32:0, {0.0082384077832102776,0.018835416063666344,0.013810183852910995,0.0082420259714126587,0.020754387602210045}>>) -> tensor<5x4xf32>
  // CHECK: %1 = "tfl.embedding_lookup"(%arg0, %0) : (tensor<2xi32>, tensor<5x4x!quant.uniform<i8:f32:0, {0.0082384077832102776,0.018835416063666344,0.013810183852910995,0.0082420259714126587,0.020754387602210045}>>) -> tensor<2x4xf32>
  %2 = "tfl.embedding_lookup"(%arg0, %1) : (tensor<2xi32>, tensor<5x4xf32>) -> tensor<2x4xf32>
  // CHECK: return %1 : tensor<2x4xf32>
  return %2 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: DQQToRequantize
func.func @DQQToRequantize(%arg0: tensor<1x128x128x320x!quant.uniform<i8:f32, 0.17072822153568268:6>>) -> (tensor<1x128x128x320x!quant.uniform<i8:f32, 0.1043805405497551:-6>>) {
    %0 = "tfl.dequantize"(%arg0) : (tensor<1x128x128x320x!quant.uniform<i8:f32, 0.17072822153568268:6>>) -> tensor<1x128x128x320xf32>
// CHECK:    %0 = "tfl.quantize"(%arg0) <{qtype = tensor<1x128x128x320x!quant.uniform<i8:f32, 0.1043805405497551:-6>>}> : (tensor<1x128x128x320x!quant.uniform<i8:f32, 0.17072822153568268:6>>) -> tensor<1x128x128x320x!quant.uniform<i8:f32, 0.1043805405497551:-6>>
    %1 = "tfl.quantize"(%0) <{qtype = tensor<1x128x128x320x!quant.uniform<i8:f32, 0.1043805405497551:-6>>}> : (tensor<1x128x128x320xf32>) -> tensor<1x128x128x320x!quant.uniform<i8:f32, 0.1043805405497551:-6>>
// CHECK:    return %0 : tensor<1x128x128x320x!quant.uniform<i8:f32, 0.1043805405497551:-6>>
    return %1 : tensor<1x128x128x320x!quant.uniform<i8:f32, 0.1043805405497551:-6>>
}

// -----

func.func @VolatileQuantizeConst() -> (tensor<1xf32>) {
  %cst = arith.constant dense<1.14751196> : tensor<1xf32>
// CHECK: %0 = "tfl.pseudo_qconst"() <{qtype = tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>, value = dense<20578> : tensor<1xi32>}> {volatile} : () -> tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>
  %0 = "tfl.quantize"(%cst) <{qtype = tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>}> {volatile} : (tensor<1xf32>) -> tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>) -> tensor<1xf32>
  %1 = "tfl.dequantize"(%0) : (tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>) -> tensor<1xf32>
// CHECK: return %1 : tensor<1xf32>
  return %1 : tensor<1xf32>
}

// -----

// CHECK-LABEL: QuantizeFloatConst
func.func @QuantizeFloatConst() -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>> {
  %0 = arith.constant dense<-0.1> : tensor<2x2xf32>
// CHECK:  %[[cst:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<0> : tensor<2x2xi8>}>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
// CHECK:  return %[[cst]]
  func.return %1 : tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
}

// -----

// CHECK-LABEL: QuantizeFloatConst4Bits
func.func @QuantizeFloatConst4Bits() -> tensor<2x4x!quant.uniform<i4:f32, 2.500000e-01:-1>> {
  %0 = arith.constant dense<[[-0.75, -0.5, -0.25, 0.0], [0.25, 0.5, 0.75, 1.0]]> : tensor<2x4xf32>
// CHECK:  %[[cst:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x4x!quant.uniform<i4:f32, 2.500000e-01:-1>>, value = dense<{{\[\[}}-4, -3, -2, -1{{\]}}, [0, 1, 2, 3{{\]\]}}> : tensor<2x4xi4>}>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x4x!quant.uniform<i4:f32, 2.500000e-01:-1>>} : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<i4:f32, 2.500000e-01:-1>>
// CHECK:  return %[[cst]]
  func.return %1 : tensor<2x4x!quant.uniform<i4:f32, 2.500000e-01:-1>>
}

// -----

// CHECK-LABEL: QuantizeDenseFloatConst
func.func @QuantizeDenseFloatConst() -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>> {
  %0 = arith.constant dense<[[-0.1, 1.0], [1.0, 3.0]]> : tensor<2x2xf32>
// CHECK:  %[[cst:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<{{\[\[}}0, -1], {{\[}}-1, -1]]> : tensor<2x2xi8>}>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
// CHECK:  return %[[cst]]
  func.return %1 : tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
}

// -----

// CHECK-LABEL: QuantizeSplatFloatConst
func.func @QuantizeSplatFloatConst() -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>> {
  %0 = arith.constant dense<3.0> : tensor<2x2xf32>
// CHECK:  %[[cst:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<-1> : tensor<2x2xi8>}>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
// CHECK:  return %[[cst]]
  func.return %1 : tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
}

// -----

// CHECK-LABEL: DequantizeAndQuantize
func.func @DequantizeAndQuantize() -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>> {
// CHECK:  %[[cst:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<-1> : tensor<2x2xi8>}>
  %cst = "tfl.pseudo_qconst"() {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<-1> : tensor<2x2xi8>} : () -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  %0 = "tfl.dequantize"(%cst) : (tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>) -> tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
// CHECK:  return %[[cst]] : tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  func.return %1 : tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
}

// -----

// CHECK-LABEL: QuantizeConv2D
func.func @QuantizeConv2D(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
// CHECK: %[[cst0:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x!quant.uniform<i32:f32, 7.812500e-04>>, value = dense<-1583> : tensor<32xi32>}>
  %0 = "tfl.quantize"(%cst) <{qtype = tensor<32x!quant.uniform<i32:f32, 7.812500e-04>>}> : (tensor<32xf32>) -> tensor<32x!quant.uniform<i32:f32, 7.812500e-04>>
  %1 = "tfl.dequantize"(%0) : (tensor<32x!quant.uniform<i32:f32, 7.812500e-04>>) -> tensor<32xf32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
  %w = arith.constant dense<-1.0> : tensor<32x3x3x3xf32>
// CHECK: %[[cst1:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 1.000000e-01>>, value = dense<1> : tensor<32x3x3x3xi8>}>
  %3 = "tfl.quantize"(%w) {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.1>>} : (tensor<32x3x3x3xf32>) -> tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.1>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.1>>) -> tensor<32x3x3x3xf32>
// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[cst1]], %[[cst0]])
  %5 = "tfl.conv_2d"(%2, %4, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
// CHECK: return %[[conv]] : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  func.return %6 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
}

// -----

// CHECK-LABEL: QuantizeConv2D4Bit
func.func @QuantizeConv2D4Bit(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
// CHECK: %[[cst0:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x!quant.uniform<i32:f32, 7.812500e-04>>, value = dense<-1583> : tensor<32xi32>}>
  %0 = "tfl.quantize"(%cst) <{qtype = tensor<32x!quant.uniform<i32:f32, 7.812500e-04>>}> : (tensor<32xf32>) -> tensor<32x!quant.uniform<i32:f32, 7.812500e-04>>
  %1 = "tfl.dequantize"(%0) : (tensor<32x!quant.uniform<i32:f32, 7.812500e-04>>) -> tensor<32xf32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
  %w = arith.constant dense<-1.0> : tensor<32x3x3x3xf32>
// CHECK: %[[cst1:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x3x3x3x!quant.uniform<u4<1:15>:f32, 1.000000e-01>>, value = dense<1> : tensor<32x3x3x3xi4>}>
  %3 = "tfl.quantize"(%w) {qtype = tensor<32x3x3x3x!quant.uniform<u4<1:15>:f32, 0.1>>} : (tensor<32x3x3x3xf32>) -> tensor<32x3x3x3x!quant.uniform<u4<1:15>:f32, 0.1>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x3x3x3x!quant.uniform<u4<1:15>:f32, 0.1>>) -> tensor<32x3x3x3xf32>
// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[cst1]], %[[cst0]])
  %5 = "tfl.conv_2d"(%2, %4, %1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
// CHECK: return %[[conv]] : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  func.return %6 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
}

// -----

// CHECK-LABEL: QuantizeDepthwiseConv2D
func.func @QuantizeDepthwiseConv2D(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
// CHECK: %[[cst0:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<-7254> : tensor<32xi32>}>
  %0 = "tfl.quantize"(%cst) <{qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>}> : (tensor<32xf32>) -> tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>
  %1 = "tfl.dequantize"(%0) : (tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>) -> tensor<32xf32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
// CHECK: %[[cst1:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>}>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>) -> tensor<32x3x3x3xf32>
// CHECK: %[[conv:.*]] = "tfl.depthwise_conv_2d"(%arg0, %[[cst1]], %[[cst0]]) <{depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32}>
  %5 = "tfl.depthwise_conv_2d"(%2, %4, %1) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
// CHECK: return %[[conv]]
  func.return %6 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
}

// -----

// CHECK-LABEL: QuantizeFullyConnected
func.func @QuantizeFullyConnected(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
// CHECK: %[[cst_0:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<-7254> : tensor<32xi32>}>
  %0 = "tfl.quantize"(%cst) <{qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>}> : (tensor<32xf32>) -> tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>
  %1 = "tfl.dequantize"(%0) : (tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>) -> tensor<32xf32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
// CHECK: %[[cst_1:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x12x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x12xi8>}>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x12x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x12xi8>} : () -> tensor<32x12x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x12x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>) -> tensor<32x12xf32>
// CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %[[cst_1]], %[[cst_0]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}>
  %5 = "tfl.fully_connected"(%2, %4, %1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x224x224x3xf32>, tensor<32x12xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
// CHECK: return %[[fc]]
  func.return %6 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
}

// -----

// CHECK-LABEL: QuantizeFullyConnected4Bit
func.func @QuantizeFullyConnected4Bit(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
// CHECK: %[[cst_0:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x!quant.uniform<i32:f32, 0.0030937367812500002>>, value = dense<-400> : tensor<32xi32>}>
  %0 = "tfl.quantize"(%cst) <{qtype = tensor<32x!quant.uniform<i32:f32, 0.0030937367812500002>>}> : (tensor<32xf32>) -> tensor<32x!quant.uniform<i32:f32, 0.0030937367812500002>>
  %1 = "tfl.dequantize"(%0) : (tensor<32x!quant.uniform<i32:f32, 0.0030937367812500002>>) -> tensor<32xf32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
// CHECK: %[[cst_1:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<32x12x!quant.uniform<u4<1:15>:f32, 0.39599830800000002:8>>, value = dense<-7> : tensor<32x12xi4>}>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x12x!quant.uniform<u4<1:15>:f32, 0.395998308:8>>, value = dense<-7> : tensor<32x12xi4>} : () -> tensor<32x12x!quant.uniform<u4<1:15>:f32, 0.395998308:8>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x12x!quant.uniform<u4<1:15>:f32, 0.395998308:8>>) -> tensor<32x12xf32>
// CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %[[cst_1]], %[[cst_0]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}>
  %5 = "tfl.fully_connected"(%2, %4, %1) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x224x224x3xf32>, tensor<32x12xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %6 = "tfl.quantize"(%5) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
// CHECK: return %[[fc]]
  func.return %6 : tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
}

// -----

// CHECK-LABEL: QuantizeNoBiasFullyConnected
func.func @QuantizeNoBiasFullyConnected(%arg0: tensor<3x!quant.uniform<u8:f32, 1.0>>, %arg1: tensor<3x3x!quant.uniform<u8<1:255>:f32, 1.0>>, %arg2: none) -> tensor<3x!quant.uniform<u8:f32, 1.0>> {
  %0 = "tfl.dequantize"(%arg0) : (tensor<3x!quant.uniform<u8:f32, 1.0>>) -> tensor<3xf32>
  %1 = "tfl.dequantize"(%arg1) : (tensor<3x3x!quant.uniform<u8<1:255>:f32, 1.0>>) -> tensor<3x3xf32>
// CHECK: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %arg2)
  %2 = "tfl.fully_connected"(%0, %1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<3xf32>, tensor<3x3xf32>, none) -> tensor<3xf32>
  %3 = "tfl.quantize"(%2) {qtype = tensor<3x!quant.uniform<u8:f32, 1.0>>} : (tensor<3xf32>) -> tensor<3x!quant.uniform<u8:f32, 1.0>>
// CHECK: return %[[fc]]
  func.return %3 : tensor<3x!quant.uniform<u8:f32, 1.0>>
}

// -----

// CHECK-LABEL: QuantizeAveragePool2D
func.func @QuantizeAveragePool2D(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16x!quant.uniform<u8:f32, 7.812500e-03:128>> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
// CHECK: %[[avgp:.*]] = "tfl.average_pool_2d"(%arg0)
  %1 = "tfl.average_pool_2d"(%0) {name = "avgpool", filter_height = 3 : i32, filter_width = 6 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 3 : i32, stride_w = 1 : i32} : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32>
  %2 = "tfl.quantize"(%1) {qtype = tensor<1x1x1x16x!quant.uniform<u8:f32, 7.812500e-03:128>>} : (tensor<1x1x1x16xf32>) -> tensor<1x1x1x16x!quant.uniform<u8:f32, 7.812500e-03:128>>
// CHECK: return %[[avgp]] : tensor<1x1x1x16x!quant.uniform<u8:f32, 7.812500e-03:128>>
  func.return %2 : tensor<1x1x1x16x!quant.uniform<u8:f32, 7.812500e-03:128>>
}

// -----

// This behavior is intentioally different than the legacy quantized pass.
// [quantized value] -> [DQ] -> [Float] pattern is no longer quantized.
// CHECK-LABEL: NoQuantizeAveragePool2D
func.func @NoQuantizeAveragePool2D(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
// CHECK: %[[DQ:.*]] = "tfl.dequantize"(%arg0)
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
// CHECK: %[[AVGP:.*]] = "tfl.average_pool_2d"(%[[DQ]])
  %1 = "tfl.average_pool_2d"(%0) {name = "avgpool", filter_height = 3 : i32, filter_width = 6 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 3 : i32, stride_w = 1 : i32} : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32>
// CHECK: return %[[AVGP]] : tensor<1x1x1x16xf32>
  func.return %1 : tensor<1x1x1x16xf32>
}

// -----

// CHECK-LABEL: QuantizeReshape2D
func.func @QuantizeReshape2D(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x36x16x!quant.uniform<u8:f32, 7.812500e-03:128>> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<[1, 36, 16]> : tensor<3xi32>
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
// CHECK: %[[rs:.*]] = "tfl.reshape"(%arg0, %{{.*}})
  %1 = "tfl.reshape"(%0, %cst) : (tensor<1x6x6x16xf32>, tensor<3xi32>) -> tensor<1x36x16xf32>
  %2 = "tfl.quantize"(%1) {qtype = tensor<1x36x16x!quant.uniform<u8:f32, 7.812500e-03:128>>} : (tensor<1x36x16xf32>) -> tensor<1x36x16x!quant.uniform<u8:f32, 7.812500e-03:128>>
// CHECK: return %[[rs]] : tensor<1x36x16x!quant.uniform<u8:f32, 7.812500e-03:128>>
  func.return %2 : tensor<1x36x16x!quant.uniform<u8:f32, 7.812500e-03:128>>
}


// -----

// CHECK-LABEL: QuantizeSoftmax
func.func @QuantizeSoftmax(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03>> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
// CHECK: %[[sm:.*]] = "tfl.softmax"(%arg0)
  %1 = "tfl.softmax"(%0) {beta = 1.000000e+00 : f32} : (tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
  %2 = "tfl.quantize"(%1) {qtype = tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03>>} : (tensor<1x6x6x16xf32>) -> tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03>>
// CHECK: return %[[sm]] : tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03>>
  func.return %2 : tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03>>
}

// -----

// CHECK-LABEL: QuantizeLogistic
func.func @QuantizeLogistic(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03>> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
// CHECK: %[[lg:.*]] = "tfl.logistic"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>)
  %1 = "tfl.logistic"(%0) : (tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
  %2 = "tfl.quantize"(%1) {qtype = tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03>>} : (tensor<1x6x6x16xf32>) -> tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03>>
// CHECK: return %[[lg]] : tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03>>
  func.return %2 : tensor<1x6x6x16x!quant.uniform<u8:f32, 3.906250e-03>>
}

// -----

// CHECK-LABEL: QuantizeAdd
func.func @QuantizeAdd(tensor<1x56x56x24x!quant.uniform<u8:f32, 0.27583434161017922:119>>, tensor<1x56x56x24x!quant.uniform<u8:f32, 0.40149296779258581:136>>) -> tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>> {
^bb0(%arg0: tensor<1x56x56x24x!quant.uniform<u8:f32, 0.27583434161017922:119>>, %arg1: tensor<1x56x56x24x!quant.uniform<u8:f32, 0.40149296779258581:136>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x56x56x24x!quant.uniform<u8:f32, 0.27583434161017922:119>>) -> tensor<1x56x56x24xf32>
  %1 = "tfl.dequantize"(%arg1) : (tensor<1x56x56x24x!quant.uniform<u8:f32, 0.40149296779258581:136>>) -> tensor<1x56x56x24xf32>
// CHECK: %[[add:.*]] = tfl.add(%arg0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<1x56x56x24x!quant.uniform<u8:f32, 0.27583434161017922:119>>, tensor<1x56x56x24x!quant.uniform<u8:f32, 0.40149296779258581:136>>)
  %2 = tfl.add %0, %1 {fused_activation_function = "NONE"} : tensor<1x56x56x24xf32> loc("Block")
  %3 = "tfl.quantize"(%2) {qtype = tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>>} : (tensor<1x56x56x24xf32>) -> tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>>
// CHECK: return %[[add]] : tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>>
  func.return %3 : tensor<1x56x56x24x!quant.uniform<u8:f32, 0.4321689530914905:133>>
}

// -----

// CHECK-LABEL: QuantizeConcat
func.func @QuantizeConcat(tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>, tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>> {
^bb0(%arg0: tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>, %arg1: tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
  %1 = "tfl.dequantize"(%arg1) : (tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %[[cc:.*]] = "tfl.concatenation"(%arg0, %arg1) <{axis = 0 : i32, fused_activation_function = "NONE"}>
  %2 = "tfl.concatenation"(%0, %1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  %3 = "tfl.quantize"(%2) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
// CHECK: return %[[cc]] : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
  func.return %3 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
}

// -----

// CHECK-LABEL: QuantizeMaxPool2D
func.func @QuantizeMaxPool2D(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16x!quant.uniform<u8:f32, 7.812500e-03:128>> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
// CHECK: %[[mp:.*]] = "tfl.max_pool_2d"(%arg0) <{filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16x!quant.uniform<u8:f32, 7.812500e-03:128>>
  %1 = "tfl.max_pool_2d"(%0) {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32>
  %2 = "tfl.quantize"(%1) {qtype = tensor<1x1x1x16x!quant.uniform<u8:f32, 7.812500e-03:128>>} : (tensor<1x1x1x16xf32>) -> tensor<1x1x1x16x!quant.uniform<u8:f32, 7.812500e-03:128>>
// CHECK: return %[[mp]] : tensor<1x1x1x16x!quant.uniform<u8:f32, 7.812500e-03:128>>
  func.return %2 : tensor<1x1x1x16x!quant.uniform<u8:f32, 7.812500e-03:128>>
}

// -----

// CHECK-LABEL: QuantizeSplit
func.func @QuantizeSplit(%arg: tensor<4x!quant.uniform<u8:f32, 1.0>>, %cst: tensor<i32>) -> (tensor<2x!quant.uniform<u8:f32, 1.0>>,tensor<2x!quant.uniform<u8:f32, 1.0>>) {
  %0 = "tfl.dequantize"(%arg) : (tensor<4x!quant.uniform<u8:f32, 1.0>>) -> tensor<4xf32>
// CHECK: %[[sp:.*]]:2 = "tfl.split"(%arg1, %arg0) <{num_splits = 2 : i32}> : (tensor<i32>, tensor<4x!quant.uniform<u8:f32, 1.000000e+00>>)
  %1:2 = "tfl.split"(%cst, %0) {num_splits = 2 : i32} : (tensor<i32>, tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>)
  %2 = "tfl.quantize"(%1#0) {qtype = tensor<2x!quant.uniform<u8:f32, 1.0>>} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 1.0>>
  %3 = "tfl.quantize"(%1#1) {qtype = tensor<2x!quant.uniform<u8:f32, 1.0>>} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 1.0>>
// CHECK: return %[[sp]]#0, %[[sp]]#1
  func.return %2, %3 : tensor<2x!quant.uniform<u8:f32, 1.0>>, tensor<2x!quant.uniform<u8:f32, 1.0>>
}

// -----

// CHECK-LABEL: QuantizeCustomTfOp
func.func @QuantizeCustomTfOp(%arg0: tensor<128x128x!quant.uniform<u8:f32, 0.1:127>>,
    %arg1: tensor<1x!quant.uniform<u8:f32, 0.2:127>>, %arg2: tensor<1x!quant.uniform<u8:f32, 0.4:127>>,
    %arg3: tensor<1xi32>) -> (tensor<128x128x!quant.uniform<u8:f32, 0.2:125>>) {
  %0 = "tfl.dequantize"(%arg0) : (tensor<128x128x!quant.uniform<u8:f32, 0.1:127>>) -> tensor<128x128xf32>
  %1 = "tfl.dequantize"(%arg1) : (tensor<1x!quant.uniform<u8:f32, 0.2:127>>) -> tensor<1xf32>
  %2 = "tfl.dequantize"(%arg2) : (tensor<1x!quant.uniform<u8:f32, 0.4:127>>) -> tensor<1xf32>
// CHECK: %4 = "tfl.custom_tf"(%arg0, %arg1, %arg2, %arg3) ({
// CHECK-NEXT: ^bb0(%arg4: tensor<128x128xf32>, %arg5: tensor<1xf32>, %arg6: tensor<1xf32>, %arg7: tensor<1xi32>):
// CHECK-NEXT:   "tf.LayerNorm"(%arg4, %arg5, %arg6, %arg7) {_tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<128x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<128x128xf32>
// CHECK-NEXT:   "tfl.yield"
// CHECK-NEXT: }) {_tfl_quant_trait = "fully_quantizable", device = ""} :
// CHECK-SAME: (tensor<128x128x!quant.uniform<u8:f32, 1.000000e-01:127>>, tensor<1x!quant.uniform<u8:f32, 2.000000e-01:127>>, tensor<1x!quant.uniform<u8:f32, 4.000000e-01:127>>, tensor<1xi32>)
// CHECK-SAME: -> tensor<128x128x!quant.uniform<u8:f32, 2.000000e-01:125>>
  %3 = "tfl.custom_tf"(%0, %1, %2, %arg3) ({
  ^bb0(%a1: tensor<128x128xf32>, %a2: tensor<1xf32>, %a3: tensor<1xf32>, %a4: tensor<1xi32>):
    %4 = "tf.LayerNorm"(%a1, %a2, %a3, %a4) {_tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<128x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<128x128xf32>
   "tfl.yield"(%4) : (tensor<128x128xf32>) -> ()
  }) {_tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<128x128xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xi32>) -> tensor<128x128xf32>
  %4 = "tfl.quantize"(%3) {qtype = tensor<128x128x!quant.uniform<u8:f32, 0.2:125>>} : (tensor<128x128xf32>) -> tensor<128x128x!quant.uniform<u8:f32, 0.2:125>>
  func.return %4 : tensor<128x128x!quant.uniform<u8:f32, 0.2:125>>
}

// -----

// Checks that legacy path correctly handles asymmetric quantized values.
// CHECK-LABEL: CheckLegacyQuantizeAdd
func.func @CheckLegacyQuantizeAdd() -> tensor<1x2x!quant.uniform<i8:f32, 0.0078431372549019607:-128>> {
  %cst = arith.constant dense<[[1.000000e+00, 2.000000e+00]]> : tensor<1x2xf32>
// CHECK:  "tfl.pseudo_qconst"() <{qtype = tensor<1x2x!quant.uniform<i8:f32, 0.0078431372549019607:-128>>, value = dense<{{\[\[}}-1, 127]]> : tensor<1x2xi8>}>
  %0 = "tfl.quantize"(%cst) {qtype = tensor<1x2x!quant.uniform<i8:f32, 0.0078431372549019607:-128>>, volatile} : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 0.0078431372549019607:-128>>
  func.return %0 : tensor<1x2x!quant.uniform<i8:f32, 0.0078431372549019607:-128>>
}

// -----

func.func private @testIfThen(tensor<*xf32>) -> tensor<*xf32>
func.func private @testIfElse(tensor<*xf32>) -> tensor<*xf32>

// CHECK-LABEL: NotQuantizeIf
func.func @NotQuantizeIf(%arg0: tensor<i1>,
                    %arg1: tensor<4x!quant.uniform<u8:f32, 1.0>>) -> (tensor<4x!quant.uniform<u8:f32, 1.0>>) {
  // CHECK: %[[dq:.*]] = "tfl.dequantize"(%arg1)
  %0 = "tfl.dequantize"(%arg1) : (tensor<4x!quant.uniform<u8:f32, 1.0>>) -> tensor<4xf32>
  // CHECK-NEXT: %[[if:.*]] = "tf.If"(%arg0, %[[dq]]
  %1 = "tf.If"(%arg0, %0) {then_branch = @testIfThen, else_branch = @testIfElse, is_stateless = false} : (tensor<i1>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%[[if]])
  %2 = "tfl.quantize"(%1) {qtype = tensor<4x!quant.uniform<u8:f32, 1.0>>} : (tensor<4xf32>) -> tensor<4x!quant.uniform<u8:f32, 1.0>>

  // CHECK-NEXT: return %[[q]]
  func.return %2 : tensor<4x!quant.uniform<u8:f32, 1.0>>
}

// -----

// CHECK-LABEL: NotQuantizeReadVariable
func.func @NotQuantizeReadVariable() -> tensor<1x2x3x!quant.uniform<u8<1:255>:f32, 0.047244094488188976:128>> {
  // CHECK: %[[handle:.*]] = "tfl.var_handle"() <{container = "", shared_name = "states"}> : () -> tensor<!tf_type.resource<tensor<1x2x3xf32>>>
  %0 = "tfl.var_handle"() {container = "", shared_name = "states"} : () -> tensor<!tf_type.resource<tensor<1x2x3xf32>>>
  // CHECK-NEXT: %[[read:.*]] = "tfl.read_variable"(%[[handle]]) : (tensor<!tf_type.resource<tensor<1x2x3xf32>>>) -> tensor<1x2x3xf32>
  %1 = "tfl.read_variable"(%0) : (tensor<!tf_type.resource<tensor<1x2x3xf32>>>) -> tensor<1x2x3xf32>
  // CHECK-NEXT: %[[quantize:.*]] = "tfl.quantize"(%[[read]]) <{qtype = tensor<1x2x3x!quant.uniform<u8<1:255>:f32, 0.047244094488188976:128>>}> : (tensor<1x2x3xf32>) -> tensor<1x2x3x!quant.uniform<u8<1:255>:f32, 0.047244094488188976:128>>
  %2 = "tfl.quantize"(%1) {qtype = tensor<1x2x3x!quant.uniform<u8<1:255>:f32, 0.047244094488188976:128>>} : (tensor<1x2x3xf32>) -> tensor<1x2x3x!quant.uniform<u8<1:255>:f32, 0.047244094488188976:128>>
  // CHECK-NEXT: return %[[quantize]]
  func.return %2 : tensor<1x2x3x!quant.uniform<u8<1:255>:f32, 0.047244094488188976:128>>
}

// -----

// CHECK-LABEL: QuantizeTposeConv
func.func @QuantizeTposeConv(%arg0: tensor<2x2x3x2048xf32>) -> tensor<2x3x2x2048x!quant.uniform<u8:f32, 0.1:128>> {
  %output_shape = arith.constant dense<[2, 3, 2, 2048]> : tensor<4xi32>
  // CHECK: %[[QARG0:.*]] = "tfl.quantize"(%arg0)
  %q_arg0 = "tfl.quantize"(%arg0) {qtype = tensor<2x2x3x2048x!quant.uniform<u8:f32, 0.1:128>>} : (tensor<2x2x3x2048xf32>) -> tensor<2x2x3x2048x!quant.uniform<u8:f32, 0.1:128>>
  %dq_arg0 = "tfl.dequantize"(%q_arg0) : (tensor<2x2x3x2048x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<2x2x3x2048xf32>
  // CHECK: %[[W:.*]] = "tfl.pseudo_qconst"()
  %q_weighs = "tfl.pseudo_qconst"() {qtype = tensor<4x2x2x2048x!quant.uniform<u8<1:255>:f32, 0.15:151>>, value = dense<-76> : tensor<4x2x2x2048xi8>} : () -> tensor<4x2x2x2048x!quant.uniform<u8<1:255>:f32, 0.15:151>>
  %dq_weights = "tfl.dequantize"(%q_weighs) : (tensor<4x2x2x2048x!quant.uniform<u8<1:255>:f32, 0.15:151>>) -> tensor<4x2x2x2048xf32>
  %bias = "tfl.no_value"() {value} : () -> none
  // CHECK: %[[CONV:.*]] = "tfl.transpose_conv"(%cst, %[[W]], %[[QARG0]], %0) <{fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<4xi32>, tensor<4x2x2x2048x!quant.uniform<u8<1:255>:f32, 1.500000e-01:151>>, tensor<2x2x3x2048x!quant.uniform<u8:f32, 1.000000e-01:128>>, none) -> tensor<2x3x2x2048x!quant.uniform<u8:f32, 1.000000e-01:128>>
  %out = "tfl.transpose_conv"(%output_shape, %dq_weights, %dq_arg0, %bias) {fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<4xi32>, tensor<4x2x2x2048xf32>, tensor<2x2x3x2048xf32>, none) -> tensor<2x3x2x2048xf32>
  %q_out = "tfl.quantize"(%out) {qtype = tensor<2x3x2x2048x!quant.uniform<u8:f32, 0.1:128>>} : (tensor<2x3x2x2048xf32>) -> tensor<2x3x2x2048x!quant.uniform<u8:f32, 0.1:128>>
  // CHECK: return %[[CONV]]
  func.return %q_out : tensor<2x3x2x2048x!quant.uniform<u8:f32, 0.1:128>>
}

// -----

// CHECK-LABEL: foldQuantWeightsIntoEmbeddingLookup
func.func @foldQuantWeightsIntoEmbeddingLookup(%arg0: tensor<3xi32>) -> tensor<3x512xf32> {
  %q_weighs = "tfl.pseudo_qconst"() {qtype = tensor<3074x512x!quant.uniform<u8<1:255>:f32, 0.15:151>>, value = dense<-76> : tensor<3074x512xi8>} : () -> tensor<3074x512x!quant.uniform<u8<1:255>:f32, 0.15:151>>
  // CHECK-NOT: "tfl.dequantize"
  %dq_weights = "tfl.dequantize"(%q_weighs) : (tensor<3074x512x!quant.uniform<u8<1:255>:f32, 0.15:151>>) -> tensor<3074x512xf32>
  // CHECK: "tfl.embedding_lookup"(%arg0, %0) : (tensor<3xi32>, tensor<3074x512x!quant.uniform<u8<1:255>:f32
  %out = "tfl.embedding_lookup"(%arg0, %dq_weights) : (tensor<3xi32>, tensor<3074x512xf32>) -> tensor<3x512xf32>
  func.return %out : tensor<3x512xf32>
}

// -----

// CHECK-LABEL: RequantizationSquash
func.func @RequantizationSquash(%arg0: tensor<64x1x128xf32>) -> tensor<64x128xf32>   {
  %cst_132 = arith.constant dense<[64, 128]> : tensor<2xi32>
  %683 = "tfl.quantize"(%arg0) <{qtype = tensor<64x1x128x!quant.uniform<i8:f32, 0.8>>}> {volatile} : (tensor<64x1x128xf32>) -> tensor<64x1x128x!quant.uniform<i8:f32, 0.8>>
  // CHECK-NOT: "tfl.dequantize"(%{{.*}}) : (tensor<64x1x128x!quant.uniform<i8:f32, 0.8>>) -> tensor<64x1x128xf32>
  %688 = "tfl.dequantize"(%683) : (tensor<64x1x128x!quant.uniform<i8:f32, 0.8>>) -> tensor<64x1x128xf32>
  %698 = "tfl.quantize"(%688) <{qtype = tensor<64x1x128x!quant.uniform<i8:f32, 0.8>>}> : (tensor<64x1x128xf32>) -> tensor<64x1x128x!quant.uniform<i8:f32, 0.8>>
  %699 = "tfl.dequantize"(%698) : (tensor<64x1x128x!quant.uniform<i8:f32, 0.8>>) -> tensor<64x1x128xf32>
  // CHECK: "tfl.reshape"
  // CHECK-SAME: (tensor<64x1x128x!quant.uniform<i8:f32, 8.000000e-01>>, tensor<2xi32>) -> tensor<64x128x!quant.uniform<i8:f32, 8.000000e-01>>
  %700 = "tfl.reshape"(%699, %cst_132) : (tensor<64x1x128xf32>, tensor<2xi32>) -> tensor<64x128xf32>
  %701 = "tfl.quantize"(%700) <{qtype = tensor<64x128x!quant.uniform<i8:f32, 0.8>>}> : (tensor<64x128xf32>) -> tensor<64x128x!quant.uniform<i8:f32, 0.8>>
  %702 = "tfl.dequantize"(%701) : (tensor<64x128x!quant.uniform<i8:f32, 0.8>>) -> tensor<64x128xf32>
  func.return %702 : tensor<64x128xf32>
}

// -----

// CHECK-LABEL: RequantizationDifferentScalesNoSquash
func.func @RequantizationDifferentScalesNoSquash(%arg0: tensor<64x1x128xf32>) -> tensor<64x128xf32>   {
  %cst_132 = arith.constant dense<[64, 128]> : tensor<2xi32>
  %683 = "tfl.quantize"(%arg0) <{qtype = tensor<64x1x128x!quant.uniform<i8:f32, 0.2>>}> {volatile} : (tensor<64x1x128xf32>) -> tensor<64x1x128x!quant.uniform<i8:f32, 0.2>>
  %688 = "tfl.dequantize"(%683) : (tensor<64x1x128x!quant.uniform<i8:f32, 0.2>>) -> tensor<64x1x128xf32>
  // CHECK: %[[REQUANT:.*]] = "tfl.quantize"(%{{.*}}) <{qtype = tensor<64x1x128x!quant.uniform<i8:f32, 8.000000e-01>>}> : (tensor<64x1x128xf32>) -> tensor<64x1x128x!quant.uniform<i8:f32, 8.000000e-01>>
  %698 = "tfl.quantize"(%688) <{qtype = tensor<64x1x128x!quant.uniform<i8:f32, 0.8>>}> : (tensor<64x1x128xf32>) -> tensor<64x1x128x!quant.uniform<i8:f32, 0.8>>
  %699 = "tfl.dequantize"(%698) : (tensor<64x1x128x!quant.uniform<i8:f32, 0.8>>) -> tensor<64x1x128xf32>
  // CHECK: "tfl.reshape"(%[[REQUANT]], %{{.*}}) : (tensor<64x1x128x!quant.uniform<i8:f32, 8.000000e-01>>, tensor<2xi32>) -> tensor<64x128x!quant.uniform<i8:f32, 8.000000e-01>>
  %700 = "tfl.reshape"(%699, %cst_132) : (tensor<64x1x128xf32>, tensor<2xi32>) -> tensor<64x128xf32>
  %701 = "tfl.quantize"(%700) <{qtype = tensor<64x128x!quant.uniform<i8:f32, 0.8>>}> : (tensor<64x128xf32>) -> tensor<64x128x!quant.uniform<i8:f32, 0.8>>
  %702 = "tfl.dequantize"(%701) : (tensor<64x128x!quant.uniform<i8:f32, 0.8>>) -> tensor<64x128xf32>
  func.return %702 : tensor<64x128xf32>
}
