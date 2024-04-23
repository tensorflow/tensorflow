// RUN: tf-opt -convert-tf-quant-types -quant-convert-tf-quant-ops-to-mhlo -canonicalize "-xla-legalize-tf=legalize-chlo=false" -split-input-file %s | FILECHECK_OPTS="" FileCheck %s


//===----------------------------------------------------------------------===//
// tf.UniformQuantizedDotHybrid legalization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @quantized_matmul_fn
func.func @quantized_matmul_fn(%input: tensor<?xf32>) -> tensor<?xf32> {
  %weight = "tf.Const"() { value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2032207D207D2074656E736F725F636F6E74656E743A20225C3030315C3030325C3030335C30303422"> : tensor<2x2x!tf_type.qint8> } : () -> tensor<2x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // CHECK: %[[CONST:.*]] = mhlo.constant()
  // CHECK-SAME{LITERAL}: value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi8>
  // CHECK-SAME: tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:3>>
  // CHECK: "mhlo.dot"(%arg0, %[[CONST]]) : (tensor<?xf32>, tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?xf32>

  %0 = "tf.UniformQuantizedDotHybrid"(%input, %weight, %weight_scales, %weight_zps) {rhs_quantization_axis = -1 : i64, rhs_quantization_min_val = -128 : i64, rhs_quantization_max_val = 127 : i64} : (tensor<?xf32>, tensor<2x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

//===----------------------------------------------------------------------===//
// tf.UniformQuantizedConvolutionHybrid legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @uniform_quantized_convolution_hybrid
func.func @uniform_quantized_convolution_hybrid(%input: tensor<1x6x6x3xf32>) -> tensor<1x4x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // CHECK: %[[CONST:.*]] = mhlo.constant()
  // CHECK-SAME{LITERAL} value = dense<127> : tensor<2x3x3x2xi8>
  // CHECK-SAME: tensor<2x3x3x2x!quant.uniform<i8:f32, 1.000000e+00:3>>
  // CHECK: mhlo.convolution(%arg0, %[[CONST]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME{LITERAL}: window = {stride = [1, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
  // CHECK-SAME{LITERAL}: batch_group_count = 1 : i64, feature_group_count = 1 : i64
  // CHECK-SAME: (tensor<1x6x6x3xf32>, tensor<2x3x3x2x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<1x4x1x2xf32>

  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "VALID",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x6x6x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x4x1x2xf32>
  func.return %0 : tensor<1x4x1x2xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantized_convolution_hybrid_same
func.func @uniform_quantized_convolution_hybrid_same(%input: tensor<1x2x2x3xf32>) -> tensor<1x2x1x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // CHECK: %[[CONST:.*]] = mhlo.constant()
  // CHECK-SAME{LITERAL} value = dense<127> : tensor<2x3x3x2xi8>
  // CHECK-SAME: tensor<2x3x3x2x!quant.uniform<i8:f32, 1.000000e+00:3>>
  // CHECK: mhlo.convolution(%arg0, %[[CONST]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME{LITERAL}: window = {stride = [1, 2], pad = [[1, 1], [1, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
  // CHECK-SAME{LITERAL}: batch_group_count = 1 : i64, feature_group_count = 1 : i64
  // CHECK-SAME: (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<1x2x1x2xf32>

  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "SAME",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x2x1x2xf32>
  func.return %0 : tensor<1x2x1x2xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantized_convolution_hybrid_explicit
func.func @uniform_quantized_convolution_hybrid_explicit(%input: tensor<1x2x2x3xf32>) -> tensor<1x3x3x2xf32> {
  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // CHECK: %[[CONST:.*]] = mhlo.constant()
  // CHECK-SAME{LITERAL} value = dense<127> : tensor<2x3x3x2xi8>
  // CHECK-SAME: tensor<2x3x3x2x!quant.uniform<i8:f32, 1.000000e+00:3>>
  // CHECK: mhlo.convolution(%arg0, %[[CONST]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME{LITERAL}: window = {stride = [1, 2], pad = [[1, 2], [3, 4]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
  // CHECK-SAME{LITERAL}: batch_group_count = 1 : i64, feature_group_count = 1 : i64
  // CHECK-SAME: (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<1x3x3x2xf32>

  %0 = "tf.UniformQuantizedConvolutionHybrid"(%input, %weight, %weight_scales, %weight_zps) {
    window_strides = [1, 2],
    padding = "EXPLICIT",
    explicit_padding = [1, 2, 3, 4],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64,
    rhs_quantization_min_val = -128 : i64,
    rhs_quantization_max_val = 127 : i64
  } : (tensor<1x2x2x3xf32>, tensor<2x3x3x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1x3x3x2xf32>
  func.return %0 : tensor<1x3x3x2xf32>
}

//===----------------------------------------------------------------------===//
// tf.UniformQuantize and tf.UniformDequantize legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @uniform_quantize_and_dequantize
func.func @uniform_quantize_and_dequantize(%arg0 : tensor<2xf32>) -> tensor<2xf32> {
  %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // CHECK: %[[QUANTIZE:.*]] = mhlo.uniform_quantize %arg0 : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 1.000000e+00:3>>
  // CHECK: %[[CONVERT_1:.*]] = mhlo.bitcast_convert %[[QUANTIZE]] : (tensor<2x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<2xi8>
  // CHECK: %[[CONVERT_2:.*]] = mhlo.bitcast_convert %[[CONVERT_1]] : (tensor<2xi8>) -> tensor<2x!quant.uniform<i8:f32, 1.000000e+00:3>>
  // CHECK: %[[DEQUANTIZE:.*]] = mhlo.uniform_dequantize %[[CONVERT_2]] : (tensor<2x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<2xf32>
  // CHECK: return %[[DEQUANTIZE]] : tensor<2xf32>

  %0 = "tf.UniformQuantize"(%arg0, %scales, %zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<2xf32>, tensor<f32>, tensor<i32>) -> tensor<2x!tf_type.qint8>
  %1 = "tf.UniformDequantize"(%0, %scales, %zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<2xf32>
  func.return %1 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_and_dequantize_per_axis
func.func @uniform_quantize_and_dequantize_per_axis(%arg0 : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %scales = "tf.Const"() { value = dense<[1.0, 2.0]> : tensor<2xf32> } : () -> tensor<2xf32>
  %zps = "tf.Const"() { value = dense<[3, 4]> : tensor<2xi32> } : () -> tensor<2xi32>

  // CHECK: %[[QUANTIZE:.*]] = mhlo.uniform_quantize %arg0 : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32:0, {1.000000e+00:3,2.000000e+00:4}>>
  // CHECK: %[[CONVERT_1:.*]] = mhlo.bitcast_convert %[[QUANTIZE]] : (tensor<2x2x!quant.uniform<i8:f32:0, {1.000000e+00:3,2.000000e+00:4}>>) -> tensor<2x2xi8>
  // CHECK: %[[CONVERT_2:.*]] = mhlo.bitcast_convert %[[CONVERT_1]] : (tensor<2x2xi8>) -> tensor<2x2x!quant.uniform<i8:f32:0, {1.000000e+00:3,2.000000e+00:4}>>
  // CHECK: %[[DEQUANTIZE:.*]] = mhlo.uniform_dequantize %[[CONVERT_2]] : (tensor<2x2x!quant.uniform<i8:f32:0, {1.000000e+00:3,2.000000e+00:4}>>) -> tensor<2x2xf32>
  // CHECK: return %[[DEQUANTIZE]] : tensor<2x2xf32>

  %0 = "tf.UniformQuantize"(%arg0, %scales, %zps) {
    quantization_axis = 0 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<2x2xf32>, tensor<2xf32>, tensor<2xi32>) -> tensor<2x2x!tf_type.qint8>
  %1 = "tf.UniformDequantize"(%0, %scales, %zps) {
    quantization_axis = 0 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<2x2x!tf_type.qint8>, tensor<2xf32>, tensor<2xi32>) -> tensor<2x2xf32>
  func.return %1 : tensor<2x2xf32>
}

//===----------------------------------------------------------------------===//
// tf.UniformRequantize legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @uniform_quantize_requantize_and_dequantize
func.func @uniform_quantize_requantize_and_dequantize(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %scales_0 = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %zps_0 = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %scales_1 = "tf.Const"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
  %zps_1 = "tf.Const"() { value = dense<5> : tensor<i32> } : () -> tensor<i32>

  // CHECK: %[[QUANTIZE:.*]] = mhlo.uniform_quantize %arg0 : (tensor<4xf32>) -> tensor<4x!quant.uniform<i8:f32, 1.000000e+00:3>>
  // CHECK: %[[CONVERT_1:.*]] = mhlo.bitcast_convert %[[QUANTIZE]] : (tensor<4x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<4xi8>
  // CHECK: %[[CONVERT_2:.*]] = mhlo.bitcast_convert %[[CONVERT_1]] : (tensor<4xi8>) -> tensor<4x!quant.uniform<i8:f32, 1.000000e+00:3>>
  // CHECK: %[[REQUANTIZE:.*]] = mhlo.uniform_quantize %[[CONVERT_2]] : (tensor<4x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<4x!quant.uniform<i8:f32, 2.000000e+00:5>>
  // CHECK: %[[CONVERT_3:.*]] = mhlo.bitcast_convert %[[REQUANTIZE]] : (tensor<4x!quant.uniform<i8:f32, 2.000000e+00:5>>) -> tensor<4xi8>
  // CHECK: %[[CONVERT_4:.*]] = mhlo.bitcast_convert %[[CONVERT_3]] : (tensor<4xi8>) -> tensor<4x!quant.uniform<i8:f32, 2.000000e+00:5>>
  // CHECK: %[[DEQUANTIZE:.*]] = mhlo.uniform_dequantize %[[CONVERT_4]] : (tensor<4x!quant.uniform<i8:f32, 2.000000e+00:5>>) -> tensor<4xf32>
  // CHECK: return %[[DEQUANTIZE]] : tensor<4xf32>

  %0 = "tf.UniformQuantize"(%arg0, %scales_0, %zps_0) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<4xf32>, tensor<f32>, tensor<i32>) -> tensor<4x!tf_type.qint8>
  %1 = "tf.UniformRequantize"(%0, %scales_0, %zps_0, %scales_1, %zps_1) {
    input_quantization_axis = -1 : i64, input_quantization_min_val = -128 : i64, input_quantization_max_val = 127 : i64,
    output_quantization_axis = -1 : i64, output_quantization_min_val = -128 : i64, output_quantization_max_val = 127 : i64
  } : (tensor<4x!tf_type.qint8>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>) -> tensor<4x!tf_type.qint8>
  %2 = "tf.UniformDequantize"(%1, %scales_1, %zps_1) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<4x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<4xf32>
  func.return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_requantize_and_dequantize_per_axis
func.func @uniform_quantize_requantize_and_dequantize_per_axis(%arg0 : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %scales_0 = "tf.Const"() { value = dense<[1.0, 2.0]> : tensor<2xf32> } : () -> tensor<2xf32>
  %zps_0 = "tf.Const"() { value = dense<[3, 4]> : tensor<2xi32> } : () -> tensor<2xi32>
  %scales_1 = "tf.Const"() { value = dense<[3.0, 4.0]> : tensor<2xf32> } : () -> tensor<2xf32>
  %zps_1 = "tf.Const"() { value = dense<[5, 6]> : tensor<2xi32> } : () -> tensor<2xi32>

  // CHECK: %[[QUANTIZE:.*]] = mhlo.uniform_quantize %arg0 : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32:0, {1.000000e+00:3,2.000000e+00:4}>>
  // CHECK: %[[CONVERT_1:.*]] = mhlo.bitcast_convert %[[QUANTIZE]] : (tensor<2x2x!quant.uniform<i8:f32:0, {1.000000e+00:3,2.000000e+00:4}>>) -> tensor<2x2xi8>
  // CHECK: %[[CONVERT_2:.*]] = mhlo.bitcast_convert %[[CONVERT_1]] : (tensor<2x2xi8>) -> tensor<2x2x!quant.uniform<i8:f32:0, {1.000000e+00:3,2.000000e+00:4}>>
  // CHECK: %[[REQUANTIZE:.*]] = mhlo.uniform_quantize %[[CONVERT_2]] : (tensor<2x2x!quant.uniform<i8:f32:0, {1.000000e+00:3,2.000000e+00:4}>>) -> tensor<2x2x!quant.uniform<i8:f32:0, {3.000000e+00:5,4.000000e+00:6}>>
  // CHECK: %[[CONVERT_3:.*]] = mhlo.bitcast_convert %[[REQUANTIZE]] : (tensor<2x2x!quant.uniform<i8:f32:0, {3.000000e+00:5,4.000000e+00:6}>>) -> tensor<2x2xi8>
  // CHECK: %[[CONVERT_4:.*]] = mhlo.bitcast_convert %[[CONVERT_3]] : (tensor<2x2xi8>) -> tensor<2x2x!quant.uniform<i8:f32:0, {3.000000e+00:5,4.000000e+00:6}>>
  // CHECK: %[[DEQUANTIZE:.*]] = mhlo.uniform_dequantize %[[CONVERT_4]] : (tensor<2x2x!quant.uniform<i8:f32:0, {3.000000e+00:5,4.000000e+00:6}>>) -> tensor<2x2xf32>
  // CHECK: return %[[DEQUANTIZE]] : tensor<2x2xf32>

  %0 = "tf.UniformQuantize"(%arg0, %scales_0, %zps_0) {
    quantization_axis = 0 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<2x2xf32>, tensor<2xf32>, tensor<2xi32>) -> tensor<2x2x!tf_type.qint8>
  %1 = "tf.UniformRequantize"(%0, %scales_0, %zps_0, %scales_1, %zps_1) {
    input_quantization_axis = 0 : i64, input_quantization_min_val = -128 : i64, input_quantization_max_val = 127 : i64,
    output_quantization_axis = 0 : i64, output_quantization_min_val = -128 : i64, output_quantization_max_val = 127 : i64
  } : (tensor<2x2x!tf_type.qint8>, tensor<2xf32>, tensor<2xi32>, tensor<2xf32>, tensor<2xi32>) -> tensor<2x2x!tf_type.qint8>
  %2 = "tf.UniformDequantize"(%1, %scales_1, %zps_1) {
    quantization_axis = 0 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<2x2x!tf_type.qint8>, tensor<2xf32>, tensor<2xi32>) -> tensor<2x2xf32>
  func.return %2 : tensor<2x2xf32>
}

//===----------------------------------------------------------------------===//
// tf.UniformQuantizedDot legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @uniform_quantized_dot
func.func @uniform_quantized_dot(%input: tensor<?xf32>) -> tensor<?x!tf_type.qint32> {
  %input_scales = "tf.Const"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
  %input_zps = "tf.Const"() { value = dense<4> : tensor<i32> } : () -> tensor<i32>

  %weight = "tf.Const"() { value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2032207D207D2074656E736F725F636F6E74656E743A20225C3030315C3030325C3030335C30303422"> : tensor<2x2x!tf_type.qint8> } : () -> tensor<2x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  %output_scales = "tf.Const"() { value = dense<3.0> : tensor<f32> } : () -> tensor<f32>
  %output_zps = "tf.Const"() { value = dense<5> : tensor<i32> } : () -> tensor<i32>

  // CHECK-DAG: %[[RHS:.*]] = mhlo.constant()
  // CHECK-SAME{LITERAL}: <{value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi8>}> : () -> tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:3>>
  // CHECK-DAG: %[[LHS:.*]] = mhlo.uniform_quantize %arg0 : (tensor<?xf32>) -> tensor<?x!quant.uniform<i8:f32, 2.000000e+00:4>>
  // CHECK-DAG: %[[CONVERT_1:.*]] = mhlo.bitcast_convert %[[LHS]] : (tensor<?x!quant.uniform<i8:f32, 2.000000e+00:4>>) -> tensor<?xi8>
  // CHECK-DAG: %[[CONVERT_2:.*]] = mhlo.bitcast_convert %[[CONVERT_1]] : (tensor<?xi8>) -> tensor<?x!quant.uniform<i8:f32, 2.000000e+00:4>>
  // CHECK: "mhlo.dot"(%[[CONVERT_2]], %[[RHS]]) : (tensor<?x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:3>>)
  // CHECK-SAME: -> tensor<?x!quant.uniform<i32:f32, 3.000000e+00:5>>

  %0 = "tf.UniformQuantize"(%input, %input_scales, %input_zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<?xf32>, tensor<f32>, tensor<i32>) -> tensor<?x!tf_type.qint8>
  %1 = "tf.UniformQuantizedDot"(
    %0, %weight,
    %input_scales, %input_zps,
    %weight_scales, %weight_zps,
    %output_scales, %output_zps) {
      lhs_quantization_axis = -1 : i64,
      lhs_quantization_min_val = -128 : i64,
      lhs_quantization_max_val = 127 : i64,
      rhs_quantization_axis = -1 : i64,
      rhs_quantization_min_val = -128 : i64,
      rhs_quantization_max_val = 127 : i64,
      output_quantization_axis = -1 : i64,
      output_quantization_min_val = -2147483648 : i64,
      output_quantization_max_val = 2147483647 : i64} : (
        tensor<?x!tf_type.qint8>, tensor<2x2x!tf_type.qint8>,
        tensor<f32>, tensor<i32>,
        tensor<f32>, tensor<i32>,
        tensor<f32>, tensor<i32>) -> tensor<?x!tf_type.qint32>
  func.return %1 : tensor<?x!tf_type.qint32>
}

//===----------------------------------------------------------------------===//
// tf.UniformQuantizedConvolution legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @uniform_quantized_convolution
func.func @uniform_quantized_convolution(%input: tensor<1x6x6x3xf32>) -> tensor<1x4x1x2x!tf_type.qint32> {
  %input_scales = "tf.Const"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
  %input_zps = "tf.Const"() { value = dense<4> : tensor<i32> } : () -> tensor<i32>

  %weight = "tf.Const"() {value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E54382074656E736F725F7368617065207B2064696D207B2073697A653A2032207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2033207D2064696D207B2073697A653A2032207D207D20696E745F76616C3A20313237"> : tensor<2x3x3x2x!tf_type.qint8>} : () -> tensor<2x3x3x2x!tf_type.qint8>
  %weight_scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %weight_zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  %output_scales = "tf.Const"() { value = dense<3.0> : tensor<f32> } : () -> tensor<f32>
  %output_zps = "tf.Const"() { value = dense<5> : tensor<i32> } : () -> tensor<i32>

  // CHECK-DAG: %[[RHS:.*]] = mhlo.constant()
  // CHECK-SAME{LITERAL}: <{value = dense<127> : tensor<2x3x3x2xi8>}> : () -> tensor<2x3x3x2x!quant.uniform<i8:f32, 1.000000e+00:3>>
  // CHECK-DAG: %[[LHS:.*]] = mhlo.uniform_quantize %arg0 : (tensor<1x6x6x3xf32>) -> tensor<1x6x6x3x!quant.uniform<i8:f32, 2.000000e+00:4>>
  // CHECK-DAG: %[[CONVERT_1:.*]] = mhlo.bitcast_convert %[[LHS]] : (tensor<1x6x6x3x!quant.uniform<i8:f32, 2.000000e+00:4>>) -> tensor<1x6x6x3xi8>
  // CHECK-DAG: %[[CONVERT_2:.*]] = mhlo.bitcast_convert %[[CONVERT_1]] : (tensor<1x6x6x3xi8>) -> tensor<1x6x6x3x!quant.uniform<i8:f32, 2.000000e+00:4>>
  // CHECK: mhlo.convolution(%[[CONVERT_2]], %[[RHS]])
  // CHECK-SAME{LITERAL}: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME{LITERAL}: window = {stride = [1, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
  // CHECK-SAME{LITERAL}: batch_group_count = 1 : i64, feature_group_count = 1 : i64
  // CHECK-SAME: (tensor<1x6x6x3x!quant.uniform<i8:f32, 2.000000e+00:4>>, tensor<2x3x3x2x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<1x4x1x2x!quant.uniform<i32:f32, 3.000000e+00:5>>

  %0 = "tf.UniformQuantize"(%input, %input_scales, %input_zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<1x6x6x3xf32>, tensor<f32>, tensor<i32>) -> tensor<1x6x6x3x!tf_type.qint8>
  %1 = "tf.UniformQuantizedConvolution"(
    %0, %weight,
    %input_scales, %input_zps,
    %weight_scales, %weight_zps,
    %output_scales, %output_zps) {
      window_strides = [1, 2],
      padding = "VALID",
      explicit_padding = [],
      lhs_dilation = [1, 1],
      rhs_dilation = [2, 2],
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
      lhs_quantization_axis = -1 : i64,
      lhs_quantization_min_val = -128 : i64,
      lhs_quantization_max_val = 127 : i64,
      rhs_quantization_axis = -1 : i64,
      rhs_quantization_min_val = -128 : i64,
      rhs_quantization_max_val = 127 : i64,
      output_quantization_axis = -1 : i64,
      output_quantization_min_val = -2147483648 : i64,
      output_quantization_max_val = 2147483647 : i64} : (
        tensor<1x6x6x3x!tf_type.qint8>, tensor<2x3x3x2x!tf_type.qint8>,
        tensor<f32>, tensor<i32>,
        tensor<f32>, tensor<i32>,
        tensor<f32>, tensor<i32>) -> tensor<1x4x1x2x!tf_type.qint32>
  func.return %1 : tensor<1x4x1x2x!tf_type.qint32>
}

//===----------------------------------------------------------------------===//
// tf.UniformQuantizedAdd legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @uniform_quantized_add
func.func @uniform_quantized_add(%arg0: tensor<3x2x!tf_type.qint32>) -> tensor<3x2x!tf_type.qint32> {
  %input_scales = "tf.Const"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
  %input_zps = "tf.Const"() { value = dense<4> : tensor<i32> } : () -> tensor<i32>
  // tensor_proto that points to dense<127> of type !tf_type.qint32.
  %bias = "tf.Const"() { value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E5433322074656E736F725F7368617065207B207D2074656E736F725F636F6E74656E743A20225C3137375C3030305C3030305C30303022"> : tensor<2x!tf_type.qint32> } : () -> tensor<2x!tf_type.qint32>
  %bias_scales = "tf.Const"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
  %bias_zps = "tf.Const"() { value = dense<4> : tensor<i32> } : () -> tensor<i32>

  %output_scales = "tf.Const"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
  %output_zps = "tf.Const"() { value = dense<4> : tensor<i32> } : () -> tensor<i32>

  // CHECK-DAG: %[[LHS:.*]] = mhlo.bitcast_convert %arg0 : (tensor<3x2xi32>) -> tensor<3x2x!quant.uniform<i32:f32, 2.000000e+00:4>>
  // CHECK-DAG: %[[RHS:.*]] = mhlo.constant() <{value = dense<127> : tensor<2xi32>}> : () -> tensor<2x!quant.uniform<i32:f32, 2.000000e+00:4>>
  // CHECK: %[[RES:.*]] = chlo.broadcast_add %[[LHS]], %[[RHS]] {broadcast_dimensions = array<i64: 1>} :
  // CHECK-SAME: (tensor<3x2x!quant.uniform<i32:f32, 2.000000e+00:4>>, tensor<2x!quant.uniform<i32:f32, 2.000000e+00:4>>)
  // CHECK-SAME: -> tensor<3x2x!quant.uniform<i32:f32, 2.000000e+00:4>>
  // CHECK: %[[RES_INT:.*]] = mhlo.bitcast_convert %[[RES]] : (tensor<3x2x!quant.uniform<i32:f32, 2.000000e+00:4>>) -> tensor<3x2xi32>
  // CHECK: return %[[RES_INT]] : tensor<3x2xi32>

  %0 = "tf.UniformQuantizedAdd"(
    %arg0, %bias,
    %input_scales, %input_zps,
    %bias_scales, %bias_zps,
    %output_scales, %output_zps) {
      lhs_quantization_axis = -1 : i64,
      lhs_quantization_min_val = -2147483648 : i64,
      lhs_quantization_max_val = 2147483647 : i64,
      rhs_quantization_axis = -1 : i64,
      rhs_quantization_min_val = -2147483648 : i64,
      rhs_quantization_max_val = 2147483647 : i64,
      output_quantization_axis = -1 : i64,
      output_quantization_min_val = -2147483648 : i64,
      output_quantization_max_val = 2147483647 : i64} : (
        tensor<3x2x!tf_type.qint32>, tensor<2x!tf_type.qint32>,
        tensor<f32>, tensor<i32>,
        tensor<f32>, tensor<i32>,
        tensor<f32>, tensor<i32>) -> tensor<3x2x!tf_type.qint32>
  func.return %0 : tensor<3x2x!tf_type.qint32>
}

//===----------------------------------------------------------------------===//
// tf.UniformQuantizedClipByValue legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @uniform_quantized_clip_by_value
func.func @uniform_quantized_clip_by_value(%input: tensor<3x2xf32>) -> tensor<3x2x!tf_type.qint32> {
  %scales = "tf.Const"() { value = dense<2.0> : tensor<2xf32> } : () -> tensor<2xf32>
  %zps = "tf.Const"() { value = dense<4> : tensor<2xi32> } : () -> tensor<2xi32>

  // tensor_proto that points to dense<127> of type !tf_type.qint32.
  // CHECK-DAG: %[[MIN_MAX:.*]] = mhlo.constant() <{value = dense<127> : tensor<2xi32>}> : () -> tensor<2x!quant.uniform<i32:f32:1, {2.000000e+00:4,2.000000e+00:4}>>
  %min = "tf.Const"() { value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E5433322074656E736F725F7368617065207B207D2074656E736F725F636F6E74656E743A20225C3137375C3030305C3030305C30303022"> : tensor<2x!tf_type.qint32> } : () -> tensor<2x!tf_type.qint32>
  %max = "tf.Const"() { value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E5433322074656E736F725F7368617065207B207D2074656E736F725F636F6E74656E743A20225C3137375C3030305C3030305C30303022"> : tensor<2x!tf_type.qint32> } : () -> tensor<2x!tf_type.qint32>

  // CHECK-DAG: %[[OPERAND:.*]] = mhlo.uniform_quantize %arg0 : (tensor<3x2xf32>) -> tensor<3x2x!quant.uniform<i32:f32:1, {2.000000e+00:4,2.000000e+00:4}>>
  %0 = "tf.UniformQuantize"(%input, %scales, %zps) {
    quantization_axis = 1 : i64, quantization_min_val = -2147483648 : i64, quantization_max_val = 2147483647 : i64
  } : (tensor<3x2xf32>, tensor<2xf32>, tensor<2xi32>) -> tensor<3x2x!tf_type.qint32>

  // CHECK-DAG: %[[CONVERT_1:.*]] = mhlo.bitcast_convert %[[OPERAND]] : (tensor<3x2x!quant.uniform<i32:f32:1, {2.000000e+00:4,2.000000e+00:4}>>) -> tensor<3x2xi32>
  // CHECK-DAG: %[[CONVERT_2:.*]] = mhlo.bitcast_convert %[[CONVERT_1]] : (tensor<3x2xi32>) -> tensor<3x2x!quant.uniform<i32:f32:1, {2.000000e+00:4,2.000000e+00:4}>>
  // CHECK: %[[MIN_CLIPPED:.*]] = chlo.broadcast_maximum %[[CONVERT_2]], %[[MIN_MAX]] {broadcast_dimensions = array<i64: 1>} :
  // CHECK-SAME: (tensor<3x2x!quant.uniform<i32:f32:1, {2.000000e+00:4,2.000000e+00:4}>>, tensor<2x!quant.uniform<i32:f32:1, {2.000000e+00:4,2.000000e+00:4}>>)
  // CHECK-SAME: -> tensor<3x2x!quant.uniform<i32:f32:1, {2.000000e+00:4,2.000000e+00:4}>>
  // CHECK: %[[MAX_CLIPPED:.*]] = chlo.broadcast_minimum %[[MIN_CLIPPED]], %[[MIN_MAX]] {broadcast_dimensions = array<i64: 1>} :
  // CHECK-SAME: (tensor<3x2x!quant.uniform<i32:f32:1, {2.000000e+00:4,2.000000e+00:4}>>, tensor<2x!quant.uniform<i32:f32:1, {2.000000e+00:4,2.000000e+00:4}>>)
  // CHECK-SAME: -> tensor<3x2x!quant.uniform<i32:f32:1, {2.000000e+00:4,2.000000e+00:4}>>
  // CHECK: %[[RESULT:.*]] = mhlo.bitcast_convert %[[MAX_CLIPPED]] : (tensor<3x2x!quant.uniform<i32:f32:1, {2.000000e+00:4,2.000000e+00:4}>>) -> tensor<3x2xi32>
  // CHECK: return %[[RESULT]] : tensor<3x2xi32>
  %1 = "tf.UniformQuantizedClipByValue"(%0, %min, %max, %scales, %zps) {
      quantization_axis = 1 : i64,
      quantization_min_val = -2147483648 : i64,
      quantization_max_val = 2147483647 : i64
  } : (tensor<3x2x!tf_type.qint32>, tensor<2x!tf_type.qint32>, tensor<2x!tf_type.qint32>, tensor<2xf32>, tensor<2xi32>) -> tensor<3x2x!tf_type.qint32>
  func.return %1 : tensor<3x2x!tf_type.qint32>
}

// -----

// CHECK-LABEL: func @uniform_quantized_clip_by_value_min_not_const
func.func @uniform_quantized_clip_by_value_min_not_const(%input: tensor<3x2x!tf_type.qint32>, %min: tensor<2x!tf_type.qint32>) -> tensor<3x2x!tf_type.qint32> {
  %scales = "tf.Const"() { value = dense<2.0> : tensor<2xf32> } : () -> tensor<2xf32>
  %zps = "tf.Const"() { value = dense<4> : tensor<2xi32> } : () -> tensor<2xi32>
  // tensor_proto that points to dense<127> of type !tf_type.qint32.
  %max = "tf.Const"() { value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E5433322074656E736F725F7368617065207B207D2074656E736F725F636F6E74656E743A20225C3137375C3030305C3030305C30303022"> : tensor<2x!tf_type.qint32> } : () -> tensor<2x!tf_type.qint32>

  // CHECK-DAG: %[[INPUT:.*]] = mhlo.bitcast_convert %arg0 : (tensor<3x2xi32>) -> tensor<3x2x!quant.uniform<i32:f32:1, {2.000000e+00:4,2.000000e+00:4}>>
  // CHECK-DAG: %[[MIN:.*]] = mhlo.bitcast_convert %arg1 : (tensor<2xi32>) -> tensor<2x!quant.uniform<i32:f32:1, {2.000000e+00:4,2.000000e+00:4}>>
  // CHECK: chlo.broadcast_maximum %[[INPUT]], %[[MIN]]
  %res = "tf.UniformQuantizedClipByValue"(%input, %min, %max, %scales, %zps) {
      quantization_axis = 1 : i64,
      quantization_min_val = -2147483648 : i64,
      quantization_max_val = 2147483647 : i64
  } : (tensor<3x2x!tf_type.qint32>, tensor<2x!tf_type.qint32>, tensor<2x!tf_type.qint32>, tensor<2xf32>, tensor<2xi32>) -> tensor<3x2x!tf_type.qint32>
  func.return %res : tensor<3x2x!tf_type.qint32>
}

// -----

// CHECK-LABEL: func @uniform_quantized_clip_by_value_max_not_const
func.func @uniform_quantized_clip_by_value_max_not_const(%input: tensor<3x2x!tf_type.qint32>, %max: tensor<2x!tf_type.qint32>) -> tensor<3x2x!tf_type.qint32> {
  %scales = "tf.Const"() { value = dense<2.0> : tensor<2xf32> } : () -> tensor<2xf32>
  %zps = "tf.Const"() { value = dense<4> : tensor<2xi32> } : () -> tensor<2xi32>
  // tensor_proto that points to dense<127> of type !tf_type.qint32.
  %min = "tf.Const"() { value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E5433322074656E736F725F7368617065207B207D2074656E736F725F636F6E74656E743A20225C3137375C3030305C3030305C30303022"> : tensor<2x!tf_type.qint32> } : () -> tensor<2x!tf_type.qint32>

  // CHECK-DAG: %[[INPUT:.*]] = mhlo.bitcast_convert %arg0 : (tensor<3x2xi32>) -> tensor<3x2x!quant.uniform<i32:f32:1, {2.000000e+00:4,2.000000e+00:4}>>
  // CHECK-DAG: %[[MAX:.*]] = mhlo.bitcast_convert %arg1 : (tensor<2xi32>) -> tensor<2x!quant.uniform<i32:f32:1, {2.000000e+00:4,2.000000e+00:4}>>
  // CHECK-DAG: %[[INPUT_1:.*]] = chlo.broadcast_maximum
  // CHECK: chlo.broadcast_minimum %[[INPUT_1]], %[[MAX]]
  %res = "tf.UniformQuantizedClipByValue"(%input, %min, %max, %scales, %zps) {
      quantization_axis = 1 : i64,
      quantization_min_val = -2147483648 : i64,
      quantization_max_val = 2147483647 : i64
  } : (tensor<3x2x!tf_type.qint32>, tensor<2x!tf_type.qint32>, tensor<2x!tf_type.qint32>, tensor<2xf32>, tensor<2xi32>) -> tensor<3x2x!tf_type.qint32>
  func.return %res : tensor<3x2x!tf_type.qint32>
}

//===----------------------------------------------------------------------===//
// quant.uniform type handling with control flow ops
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @while_region_with_quant
func.func @while_region_with_quant(%arg0: tensor<?xf32>, %arg1: tensor<i32>) -> tensor<?xf32> {
  %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %one = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>

  // CHECK: %[[QUANT0:.*]] = mhlo.uniform_quantize %[[ARG:.*]] : (tensor<?xf32>) -> tensor<?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  // CHECK: %[[CONVERT_1:.*]] = mhlo.bitcast_convert %[[QUANT0]] : (tensor<?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?xi8>
  // CHECK: mhlo.while()
  // CHECK: cond
  // CHECK: %[[CHECK_RES:.*]] = chlo.broadcast_compare
  // CHECK: mhlo.return %[[CHECK_RES]] : tensor<i1>
  // CHECK: %[[CONVERT_2:.*]] = mhlo.bitcast_convert %[[CONVERT_1]] : (tensor<?xi8>) -> tensor<?x!quant.uniform<i8:f32, 1.000000e+00:3>>
  // CHECK: %[[RET:.*]] = mhlo.uniform_dequantize %[[CONVERT_2]] : (tensor<?x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<?xf32>
  // CHECK: return %[[RET]] : tensor<?xf32>

  %0 = "tf.UniformQuantize"(%arg0, %scales, %zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<?xf32>, tensor<f32>, tensor<i32>) -> tensor<?x!tf_type.qint8>
  %1 = "tf.WhileRegion"(%0) ({
  ^bb0(%carg0: tensor<?x!tf_type.qint8>):
    %check = "tf.Equal"(%arg1, %one) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "tf.Yield"(%check) : (tensor<i1>) -> ()
    }, {
  ^bb0(%barg0: tensor<?x!tf_type.qint8>):
    %id = "tf.Identity"(%barg0) : (tensor<?x!tf_type.qint8>) -> tensor<?x!tf_type.qint8>
    "tf.Yield"(%id) : (tensor<?x!tf_type.qint8>) -> ()
  }) {is_stateless = false} : (tensor<?x!tf_type.qint8>) -> tensor<?x!tf_type.qint8>
  %2 = "tf.UniformDequantize"(%1, %scales, %zps) {quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64} : (tensor<?x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<?xf32>
  func.return %2 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @while_region_with_quant_two_args
func.func @while_region_with_quant_two_args(%arg0: tensor<2x2xf32>, %arg1: tensor<i32>) -> (tensor<2x?xf32>, tensor<?x2xf32>) {
  %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %zps2 = "tf.Const"() { value = dense<2> : tensor<i32> } : () -> tensor<i32>
  %zps4 = "tf.Const"() { value = dense<4> : tensor<i32> } : () -> tensor<i32>
  %one = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>

  // CHECK: %[[QUANT0:.*]] = mhlo.uniform_quantize %[[ARG:.*]] : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:2>>
  // CHECK: %[[INT0:.*]] = mhlo.bitcast_convert %[[QUANT0]] : (tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:2>>) -> tensor<2x2xi8>
  %0 = "tf.UniformQuantize"(%arg0, %scales, %zps2) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<2x2xf32>, tensor<f32>, tensor<i32>) -> tensor<2x2x!tf_type.qint8>
  // CHECK: %[[QUANT1:.*]] = mhlo.uniform_quantize %[[ARG:.*]] : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:4>>
  // CHECK: %[[INT1:.*]] = mhlo.bitcast_convert %[[QUANT1]] : (tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00:4>>) -> tensor<2x2xi8>
  %1 = "tf.UniformQuantize"(%arg0, %scales, %zps4) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<2x2xf32>, tensor<f32>, tensor<i32>) -> tensor<2x2x!tf_type.qint8>

  // CHECK: %[[WHILE_RESULT:.*]]:2 = mhlo.while(%[[ARG0:.*]] = %[[INT0]], %[[ARG1:.*]] = %[[INT1]])
  // CHECK-SAME: tensor<2x2xi8>, tensor<2x2xi8>

  // CHECK: cond

  // CHECK: do
  // CHECK: mhlo.return %[[ARG0]], %[[ARG1]] : tensor<2x?xi8>, tensor<?x2xi8>

  %2:2 = "tf.WhileRegion"(%0, %1) ({
  ^bb0(%carg0: tensor<2x2x!tf_type.qint8>, %carg1: tensor<2x2x!tf_type.qint8>):
    %check = "tf.Equal"(%arg1, %one) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "tf.Yield"(%check) : (tensor<i1>) -> ()
    }, {
  ^bb0(%barg0: tensor<2x?x!tf_type.qint8>, %barg1: tensor<?x2x!tf_type.qint8>):
    %id = "tf.Identity"(%barg0) : (tensor<2x?x!tf_type.qint8>) -> tensor<2x?x!tf_type.qint8>
    "tf.Yield"(%id, %barg1) : (tensor<2x?x!tf_type.qint8>, tensor<?x2x!tf_type.qint8>) -> ()
  }) {is_stateless = false} : (tensor<2x2x!tf_type.qint8>, tensor<2x2x!tf_type.qint8>) -> (tensor<2x?x!tf_type.qint8>, tensor<?x2x!tf_type.qint8>)

  // %[[RESULT0:.*]] = mhlo.uniform_dequantize %[[WHILE_RESULT]]#0
  %3 = "tf.UniformDequantize"(%2#0, %scales, %zps2) {quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64} : (tensor<2x?x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<2x?xf32>

  // %[[RESULT1:.*]] = mhlo.uniform_dequantize %[[WHILE_RESULT]]#0
  %4 = "tf.UniformDequantize"(%2#1, %scales, %zps4) {quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64} : (tensor<?x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<?x2xf32>

  // return %[[RESULT0]], %[[RESULT1]]
  func.return %3, %4 : tensor<2x?xf32>, tensor<?x2xf32>
}
