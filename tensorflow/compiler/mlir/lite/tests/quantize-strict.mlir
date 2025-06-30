// RUN: litert-opt %s -tfl-quantize='qdq-conversion-mode=Strict' | FileCheck %s
// CHECK-LABEL: QuantizeConvDRQ
func.func private @XlaCallModule_quant.fake_quant.impl_0(%arg0: tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
func.func @QuantizeConvDRQ(%arg0: tensor<1x4x4x3xf32>) -> (tensor<1x4x4x1xf32>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<1xf32>
  %cst_0 = arith.constant dense<[[[[1.76285899, -0.257785767, 0.20429258], [1.16310906, 0.23124367, 0.529797196]], [[0.348971426, -0.319283515, -0.772461354], [0.316666812, 1.88180697, -1.78054631]]]]> : tensor<1x2x2x3xf32>
  %0 = stablehlo.composite "quant.fake_quant" %arg0 {composite_attributes = {dtype = "i8", narrow_range = false, quantization_dimension = 0 : i32, scale = dense<> : tensor<0xf64>, zero_point = dense<> : tensor<0xi64>}, decomposition = @XlaCallModule_quant.fake_quant.impl_0} : (tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
  %1 = "tfl.quantize"(%cst_0) <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>}> : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
  %2 = "tfl.dequantize"(%1) : (tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>) -> tensor<1x2x2x3xf32>
  %3 = "tfl.conv_2d"(%0, %2, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x4x4x3xf32>, tensor<1x2x2x3xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  return %3 : tensor<1x4x4x1xf32>

// CHECK:    %cst = arith.constant dense<0.000000e+00> : tensor<1xf32>
// CHECK{LITERAL}:    %0 = "tfl.pseudo_qconst"() <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>, value = dense<[[[[119, -17, 14], [78, 16, 36]], [[24, -22, -52], [21, 127, -120]]]]> : tensor<1x2x2x3xi8>}> : () -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
// CHECK:    %1 = "tfl.conv_2d"(%arg0, %0, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x4x4x3xf32>, tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
// CHECK:    return %1 : tensor<1x4x4x1xf32>
}

// -----

// CHECK-LABEL: QuantizeConvWithBiasDRQ
func.func @QuantizeConvWithBiasDRQ(%arg0: tensor<1x4x4x3xf32>) -> (tensor<1x4x4x1xf32>) {
  %cst = arith.constant dense<1.14751196> : tensor<1xf32>
  %cst_0 = arith.constant dense<[[[[1.76285899, -0.257785767, 0.20429258], [1.16310906, 0.23124367, 0.529797196]], [[0.348971426, -0.319283515, -0.772461354], [0.316666812, 1.88180697, -1.78054631]]]]> : tensor<1x2x2x3xf32>
  %0 = stablehlo.composite "quant.fake_quant" %arg0 {composite_attributes = {dtype = "i8", narrow_range = false, quantization_dimension = 0 : i32, scale = dense<> : tensor<0xf64>, zero_point = dense<> : tensor<0xi64>}, decomposition = @XlaCallModule_quant.fake_quant.impl_0} : (tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
  %1 = "tfl.quantize"(%cst_0) <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>}> : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
  %2 = "tfl.dequantize"(%1) : (tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>) -> tensor<1x2x2x3xf32>
  %3 = "tfl.conv_2d"(%0, %2, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x4x4x3xf32>, tensor<1x2x2x3xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  return %3 : tensor<1x4x4x1xf32>

// CHECK:    %cst = arith.constant dense<1.14751196> : tensor<1xf32>
// CHECK{LITERAL}:    %0 = "tfl.pseudo_qconst"() <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>, value = dense<[[[[119, -17, 14], [78, 16, 36]], [[24, -22, -52], [21, 127, -120]]]]> : tensor<1x2x2x3xi8>}> : () -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
// CHECK:    %1 = "tfl.conv_2d"(%arg0, %0, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x4x4x3xf32>, tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
// CHECK:    return %1 : tensor<1x4x4x1xf32>
}

// -----

// CHECK-LABEL: QuantizeConvWithBiasAndReluDRQ
func.func @QuantizeConvWithBiasAndReluDRQ(%arg0: tensor<1x4x4x3xf32>) -> (tensor<1x4x4x1xf32>) {
  %cst = arith.constant dense<1.14751196> : tensor<1xf32>
  %cst_0 = arith.constant dense<[[[[1.76285899, -0.257785767, 0.20429258], [1.16310906, 0.23124367, 0.529797196]], [[0.348971426, -0.319283515, -0.772461354], [0.316666812, 1.88180697, -1.78054631]]]]> : tensor<1x2x2x3xf32>
  %0 = stablehlo.composite "quant.fake_quant" %arg0 {composite_attributes = {dtype = "i8", narrow_range = false, quantization_dimension = 0 : i32, scale = dense<> : tensor<0xf64>, zero_point = dense<> : tensor<0xi64>}, decomposition = @XlaCallModule_quant.fake_quant.impl_0} : (tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
  %1 = "tfl.quantize"(%cst_0) <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>}> : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
  %2 = "tfl.dequantize"(%1) : (tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>) -> tensor<1x2x2x3xf32>
  %3 = "tfl.conv_2d"(%0, %2, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x4x4x3xf32>, tensor<1x2x2x3xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  return %3 : tensor<1x4x4x1xf32>

// CHECK: %cst = arith.constant dense<1.14751196> : tensor<1xf32>
// CHECK{LITERAL}: %0 = "tfl.pseudo_qconst"() <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>, value = dense<[[[[119, -17, 14], [78, 16, 36]], [[24, -22, -52], [21, 127, -120]]]]> : tensor<1x2x2x3xi8>}> : () -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
// CHECK: %1 = "tfl.conv_2d"(%arg0, %0, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x4x4x3xf32>, tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
// CHECK: return %1 : tensor<1x4x4x1xf32>
}

// -----

// CHECK-LABEL: QuantizeConvWithBiasAndReluWeightOnly
func.func @QuantizeConvWithBiasAndReluWeightOnly(%arg0: tensor<1x4x4x3xf32>) -> (tensor<1x4x4x1xf32>) {
  %cst = arith.constant dense<1.14751196> : tensor<1xf32>
  %cst_0 = arith.constant dense<[[[[1.76285899, -0.257785767, 0.20429258], [1.16310906, 0.23124367, 0.529797196]], [[0.348971426, -0.319283515, -0.772461354], [0.316666812, 1.88180697, -1.78054631]]]]> : tensor<1x2x2x3xf32>
  %0 = "tfl.quantize"(%cst_0) <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>}> : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
  %1 = "tfl.dequantize"(%0) : (tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>) -> tensor<1x2x2x3xf32>
  %2 = "tfl.conv_2d"(%arg0, %1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x4x4x3xf32>, tensor<1x2x2x3xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  return %2 : tensor<1x4x4x1xf32>

// CHECK:  %cst = arith.constant dense<1.14751196> : tensor<1xf32>
// CHECK{LITERAL}:  %0 = "tfl.pseudo_qconst"() <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>, value = dense<[[[[119, -17, 14], [78, 16, 36]], [[24, -22, -52], [21, 127, -120]]]]> : tensor<1x2x2x3xi8>}> : () -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
// CHECK:  %1 = "tfl.dequantize"(%0) : (tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>) -> tensor<1x2x2x3xf32>
// CHECK:  %2 = "tfl.conv_2d"(%arg0, %1, %cst) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x4x4x3xf32>, tensor<1x2x2x3xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
// CHECK:  return %2 : tensor<1x4x4x1xf32>
}

// -----

// CHECK-LABEL: QuantizeConvWithBiasAndReluSRQ
func.func @QuantizeConvWithBiasAndReluSRQ(%arg0: tensor<1x4x4x3xf32>) -> (tensor<1x4x4x1xf32>) {
  %cst = arith.constant dense<1.14751196> : tensor<1xf32>
  %0 = "tfl.quantize"(%cst) <{qtype = tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>}> : (tensor<1xf32>) -> tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>
  %1 = "tfl.dequantize"(%0) : (tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>) -> tensor<1xf32>
  %cst_0 = arith.constant dense<[[[[1.76285899, -0.257785767, 0.20429258], [1.16310906, 0.23124367, 0.529797196]], [[0.348971426, -0.319283515, -0.772461354], [0.316666812, 1.88180697, -1.78054631]]]]> : tensor<1x2x2x3xf32>
  %2 = "tfl.quantize"(%arg0) <{qtype = tensor<1x4x4x3x!quant.uniform<i8:f32, 0.0037634586915373802:-128>>}> : (tensor<1x4x4x3xf32>) -> tensor<1x4x4x3x!quant.uniform<i8:f32, 0.0037634586915373802:-128>>
  %3 = "tfl.dequantize"(%2) : (tensor<1x4x4x3x!quant.uniform<i8:f32, 0.0037634586915373802:-128>>) -> tensor<1x4x4x3xf32>
  %4 = "tfl.quantize"(%cst_0) <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>}> : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
  %5 = "tfl.dequantize"(%4) : (tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>) -> tensor<1x2x2x3xf32>
  %6 = "tfl.conv_2d"(%3, %5, %1) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x4x4x3xf32>, tensor<1x2x2x3xf32>, tensor<1xf32>) -> tensor<1x4x4x1xf32>
  %7 = "tfl.quantize"(%6) <{qtype = tensor<1x4x4x1x!quant.uniform<i8:f32, 0.013401651754975319:-128>>}> : (tensor<1x4x4x1xf32>) -> tensor<1x4x4x1x!quant.uniform<i8:f32, 0.013401651754975319:-128>>
  %8 = "tfl.dequantize"(%7) : (tensor<1x4x4x1x!quant.uniform<i8:f32, 0.013401651754975319:-128>>) -> tensor<1x4x4x1xf32>
  return %8 : tensor<1x4x4x1xf32>

// CHECK: %0 = "tfl.pseudo_qconst"() <{qtype = tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>, value = dense<20578> : tensor<1xi32>}> : () -> tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>
// CHECK: %1 = "tfl.quantize"(%arg0) <{qtype = tensor<1x4x4x3x!quant.uniform<i8:f32, 0.0037634586915373802:-128>>}> : (tensor<1x4x4x3xf32>) -> tensor<1x4x4x3x!quant.uniform<i8:f32, 0.0037634586915373802:-128>>
// CHECK{LITERAL}: %2 = "tfl.pseudo_qconst"() <{qtype = tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>, value = dense<[[[[119, -17, 14], [78, 16, 36]], [[24, -22, -52], [21, 127, -120]]]]> : tensor<1x2x2x3xi8>}> : () -> tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>
// CHECK: %3 = "tfl.conv_2d"(%1, %2, %0) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x4x4x3x!quant.uniform<i8:f32, 0.0037634586915373802:-128>>, tensor<1x2x2x3x!quant.uniform<i8:f32, 0.014817377552390099>>, tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>) -> tensor<1x4x4x1x!quant.uniform<i8:f32, 0.013401651754975319:-128>>
// CHECK: %4 = "tfl.dequantize"(%3) : (tensor<1x4x4x1x!quant.uniform<i8:f32, 0.013401651754975319:-128>>) -> tensor<1x4x4x1xf32>
// CHECK: return %4 : tensor<1x4x4x1xf32>
}

// -----

// CHECK-LABEL: QuantizeEmbeddingLookupDrq
func.func @QuantizeEmbeddingLookupDrq(%arg0: tensor<2xi32>) -> (tensor<2x4xf32>){
  %cst = arith.constant dense<[[1.0545162, -0.969288647, -0.594602108, -0.0318857245], [2.41093326, -1.87844908, -0.784769594, -0.313708425], [0.333708912, 1.76770353, -1.02776456, 1.41117179], [-0.508497119, -0.526377499, 0.503150403, 1.05497932], [-0.0874073281, 0.795816719, 2.65656161, -0.58229059]]> : tensor<5x4xf32>
  %0 = "tfl.quantize"(%cst) <{qtype = tensor<5x4x!quant.uniform<i8:f32:0, {0.0082384077832102776,0.018835416063666344,0.013810183852910995,0.0082420259714126587,0.020754387602210045}>>}> : (tensor<5x4xf32>) -> tensor<5x4x!quant.uniform<i8:f32:0, {0.0082384077832102776,0.018835416063666344,0.013810183852910995,0.0082420259714126587,0.020754387602210045}>>
  %1 = "tfl.dequantize"(%0) : (tensor<5x4x!quant.uniform<i8:f32:0, {0.0082384077832102776,0.018835416063666344,0.013810183852910995,0.0082420259714126587,0.020754387602210045}>>) -> tensor<5x4xf32>
  %2 = "tfl.embedding_lookup"(%arg0, %1) : (tensor<2xi32>, tensor<5x4xf32>) -> tensor<2x4xf32>
  return %2 : tensor<2x4xf32>

// CHECK{LITERAL}: %0 = "tfl.pseudo_qconst"() <{qtype = tensor<5x4x!quant.uniform<i8:f32:0, {0.0082384077832102776,0.018835416063666344,0.013810183852910995,0.0082420259714126587,0.020754387602210045}>>, value = dense<[[127, -118, -72, -4], [127, -100, -42, -17], [24, 127, -74, 102], [-62, -64, 61, 127], [-4, 38, 127, -28]]> : tensor<5x4xi8>}> : () -> tensor<5x4x!quant.uniform<i8:f32:0, {0.0082384077832102776,0.018835416063666344,0.013810183852910995,0.0082420259714126587,0.020754387602210045}>>
// CHECK: %1 = "tfl.embedding_lookup"(%arg0, %0) : (tensor<2xi32>, tensor<5x4x!quant.uniform<i8:f32:0, {0.0082384077832102776,0.018835416063666344,0.013810183852910995,0.0082420259714126587,0.020754387602210045}>>) -> tensor<2x4xf32>
// CHECK: return %1 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: DQQToRequantize
func.func @DQQToRequantize(%arg0: tensor<1x128x128x320x!quant.uniform<i8:f32, 0.17072822153568268:6>>) -> (tensor<1x128x128x320x!quant.uniform<i8:f32, 0.1043805405497551:-6>>) {
    %0 = "tfl.dequantize"(%arg0) : (tensor<1x128x128x320x!quant.uniform<i8:f32, 0.17072822153568268:6>>) -> tensor<1x128x128x320xf32>
    %1 = "tfl.quantize"(%0) <{qtype = tensor<1x128x128x320x!quant.uniform<i8:f32, 0.1043805405497551:-6>>}> : (tensor<1x128x128x320xf32>) -> tensor<1x128x128x320x!quant.uniform<i8:f32, 0.1043805405497551:-6>>
    return %1 : tensor<1x128x128x320x!quant.uniform<i8:f32, 0.1043805405497551:-6>>

// CHECK:    %0 = "tfl.quantize"(%arg0) <{qtype = tensor<1x128x128x320x!quant.uniform<i8:f32, 0.1043805405497551:-6>>}> : (tensor<1x128x128x320x!quant.uniform<i8:f32, 0.17072822153568268:6>>) -> tensor<1x128x128x320x!quant.uniform<i8:f32, 0.1043805405497551:-6>>
// CHECK:    return %0 : tensor<1x128x128x320x!quant.uniform<i8:f32, 0.1043805405497551:-6>>
}

// -----

func.func @VolatileQuantizeConst() -> (tensor<1xf32>) {
  %cst = arith.constant dense<1.14751196> : tensor<1xf32>
  %0 = "tfl.quantize"(%cst) <{qtype = tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>}> {volatile} : (tensor<1xf32>) -> tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>
  %1 = "tfl.dequantize"(%0) : (tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>) -> tensor<1xf32>
  return %1 : tensor<1xf32>
// CHECK: %0 = "tfl.pseudo_qconst"() <{qtype = tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>, value = dense<20578> : tensor<1xi32>}> {volatile} : () -> tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<1x!quant.uniform<i32:f32, 5.576458833533339E-5>>) -> tensor<1xf32>
// CHECK: return %1 : tensor<1xf32>
}
