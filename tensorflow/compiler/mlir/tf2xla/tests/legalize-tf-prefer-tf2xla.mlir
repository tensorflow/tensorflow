// RUN: tf-opt "-xla-legalize-tf=device-type=XLA_CPU_JIT legalize-chlo=false use-tf2xla-fallback=true prefer-tf2xla=true" %s | FileCheck %s
// RUN: tf-opt "-xla-legalize-tf=device-type=XLA_CPU_JIT legalize-chlo=false prefer-tf2xla=true" %s | FileCheck --check-prefix NOFALLBACK %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {

// CHECK-LABEL: @abs
func.func @abs(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-NOT: tf.Abs
  %0 = "tf.Abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @testBroadcastGradientArgs
func.func @testBroadcastGradientArgs(%s0: tensor<4xi32>, %s1: tensor<4xi32>) -> (tensor<1xi32>, tensor<0xi32>) {
  // CHECK:     tf.BroadcastGradientArgs
  %r0, %r1 = "tf.BroadcastGradientArgs"(%s0, %s1) : (tensor<4xi32>, tensor<4xi32>) -> (tensor<1xi32>, tensor<0xi32>)
  func.return %r0, %r1 : tensor<1xi32>, tensor<0xi32>
}

// -----

// CHECK-LABEL: @acos
func.func @acos(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-NOT:  tf.Acos
  %0 = "tf.Acos"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// NOFALLBACK-LABEL: @xla_svd
func.func @xla_svd(%arg0: tensor<1x1xf32>) -> (tensor<1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) {
  // NOFALLBACK: XlaSvd
  %s, %u, %v = "tf.XlaSvd"(%arg0) {max_iter = 1, epsilon = 1.0E-09 : f32, precision_config = ""} : (tensor<1x1xf32>) -> (tensor<1xf32>, tensor<1x1xf32>, tensor<1x1xf32>)
  func.return %s, %u, %v : tensor<1xf32>, tensor<1x1xf32>, tensor<1x1xf32>
}

//===----------------------------------------------------------------------===//
// Random op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @random_uniform_simple
func.func @random_uniform_simple(%arg0: tensor<3xi32>) -> tensor<12x?x64xf32> {
  // expected-remark@+1 {{lowering requires operand #0 to be a constant}}
  %0 = "tf.RandomUniform"(%arg0) {device = "", seed = 0 : i64, seed2 = 0 : i64} : (tensor<3xi32>) -> tensor<12x?x64xf32>
  func.return %0 : tensor<12x?x64xf32>
}

// -----
// CHECK-LABEL: func @random_uniform_with_seeds
func.func @random_uniform_with_seeds(%arg0: tensor<4xi32>) -> tensor<32x12x12x64xf32> {
    // CHECK: %0 = mhlo.constant dense<[32, 12, 12, 64]> : tensor<4xi32>
    // CHECK-NEXT: %1 = mhlo.constant dense<[32, 12, 12, 64]> : tensor<4xi32>
    // CHECK-NEXT: %2 = "mhlo.slice"(%1) <{limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<4xi32>) -> tensor<1xi32>
    // CHECK-NEXT: %3 = mhlo.reshape %2 : (tensor<1xi32>) -> tensor<i32>
    // CHECK-NEXT: %4 = mhlo.convert %3 : tensor<i32>
    // CHECK-NEXT: %5 = "mhlo.slice"(%1) <{limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<4xi32>) -> tensor<1xi32>
    // CHECK-NEXT: %6 = mhlo.reshape %5 : (tensor<1xi32>) -> tensor<i32>
    // CHECK-NEXT: %7 = mhlo.convert %6 : tensor<i32>
    // CHECK-NEXT: %8 = "mhlo.slice"(%1) <{limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<4xi32>) -> tensor<1xi32>
    // CHECK-NEXT: %9 = mhlo.reshape %8 : (tensor<1xi32>) -> tensor<i32>
    // CHECK-NEXT: %10 = mhlo.convert %9 : tensor<i32>
    // CHECK-NEXT: %11 = "mhlo.slice"(%1) <{limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<4xi32>) -> tensor<1xi32>
    // CHECK-NEXT: %12 = mhlo.reshape %11 : (tensor<1xi32>) -> tensor<i32>
    // CHECK-NEXT: %13 = mhlo.convert %12 : tensor<i32>
    // CHECK-NEXT: %14 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    // CHECK-NEXT: %15 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    // CHECK-NEXT: %16 = mhlo.constant dense<[32, 12, 12, 64]> : tensor<4xi64>
    // CHECK-NEXT: %17 = "mhlo.rng"(%14, %15, %16) <{rng_distribution = #mhlo.rng_distribution<UNIFORM>}> : (tensor<f32>, tensor<f32>, tensor<4xi64>) -> tensor<32x12x12x64xf32>
  %cst = "tf.Const"() {value = dense<[32, 12, 12, 64]> : tensor<4xi32>} : () -> tensor<4xi32>
  %0 = "tf.RandomUniform"(%cst) {seed = 87654321 : i64, seed2 = 0 : i64} : (tensor<4xi32>) -> tensor<32x12x12x64xf32>
    // CHECK: return %17 : tensor<32x12x12x64xf32>
  func.return %0 : tensor<32x12x12x64xf32>
}

// CHECK-LABEL: slice_variable_start
func.func @slice_variable_start(%arg0: tensor<3x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  // CHECK: %[[SLICED_START1:.*]] = "mhlo.slice"(%arg1)
  // CHECK-DAG-SAME: {limit_indices = dense<1> : tensor<1xi64>,
  // CHECK-DAG-SAME: start_indices = dense<0> : tensor<1xi64>,
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK: %[[RESHAPED_START1:.*]] = mhlo.reshape %[[SLICED_START1]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[SLICED_START2:.*]] = "mhlo.slice"(%arg1)
  // CHECK-DAG-SAME: {limit_indices = dense<2> : tensor<1xi64>,
  // CHECK-DAG-SAME: start_indices = dense<1> : tensor<1xi64>,
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK: %[[RESHAPED_START2:.*]] = mhlo.reshape %[[SLICED_START2]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[RESULT:.*]] = "mhlo.dynamic_slice"(%arg0, %[[RESHAPED_START1]], %[[RESHAPED_START2]]) <{slice_sizes = dense<[1, 4]> : tensor<2xi64>}> : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  // CHECK: return %[[RESULT]] : tensor<1x4xi32>
  %sizes = "tf.Const"() {value = dense<[1, 4]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  %0 = "tf.Slice"(%arg0, %arg1, %sizes) : (tensor<3x4xi32>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// CHECK-LABEL: slice_variable_start_negsize
func.func @slice_variable_start_negsize(%arg0: tensor<3x4xi32>, %arg1: tensor<2xi32>) -> tensor<1x4xi32> {
  // CHECK: %0 = mhlo.constant dense<[1, -1]> : tensor<2xi32>
  // CHECK-NEXT: %1 = mhlo.constant dense<[1, -1]> : tensor<2xi32>
  // CHECK-NEXT: %2 = "mhlo.slice"(%1) <{limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %3 = mhlo.reshape %2 : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %4 = mhlo.constant dense<[1, -1]> : tensor<2xi32>
  // CHECK-NEXT: %5 = "mhlo.slice"(%4) <{limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %6 = mhlo.reshape %5 : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %7 = mhlo.constant dense<3> : tensor<i32>
  // CHECK-NEXT: %8 = "mhlo.set_dimension_size"(%arg0, %7) <{dimension = 0 : i64}> : (tensor<3x4xi32>, tensor<i32>) -> tensor<3x4xi32>
  // CHECK-NEXT: %9 = mhlo.constant dense<4> : tensor<i32>
  // CHECK-NEXT: %10 = "mhlo.set_dimension_size"(%8, %9) <{dimension = 1 : i64}> : (tensor<3x4xi32>, tensor<i32>) -> tensor<3x4xi32>
  // CHECK-NEXT: %11 = mhlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %12 = "mhlo.pad"(%10, %11) <{edge_padding_high = dense<[3, 4]> : tensor<2xi64>, edge_padding_low = dense<0> : tensor<2xi64>, interior_padding = dense<0> : tensor<2xi64>}> : (tensor<3x4xi32>, tensor<i32>) -> tensor<6x8xi32>
  // CHECK-NEXT: %13 = "mhlo.slice"(%arg1) <{limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %14 = mhlo.reshape %13 : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %15 = "mhlo.slice"(%arg1) <{limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %16 = mhlo.reshape %15 : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %17 = "mhlo.dynamic_slice"(%12, %14, %16) <{slice_sizes = dense<[3, 4]> : tensor<2xi64>}> : (tensor<6x8xi32>, tensor<i32>, tensor<i32>) -> tensor<3x4xi32>
  // CHECK-NEXT: %18 = "mhlo.slice"(%17) <{limit_indices = dense<[1, 4]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}> : (tensor<3x4xi32>) -> tensor<1x4xi32>
  // CHECK-NEXT: %19 = mhlo.constant dense<4> : tensor<i32>
  // CHECK-NEXT: %20 = mhlo.subtract %19, %16 : tensor<i32>
  // CHECK-NEXT: %21 = "mhlo.set_dimension_size"(%18, %20) <{dimension = 1 : i64}> : (tensor<1x4xi32>, tensor<i32>) -> tensor<1x?xi32, #mhlo.type_extensions<bounds = [?, 4]>>
  // CHECK-NEXT: %cast = tensor.cast %21 : tensor<1x?xi32, #mhlo.type_extensions<bounds = [?, 4]>> to tensor<1x4xi32>
  // CHECK-NEXT: return %cast : tensor<1x4xi32>
  %sizes = "tf.Const"() {value = dense<[1, -1]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %0 = "tf.Slice"(%arg0, %arg1, %sizes) : (tensor<3x4xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

//===----------------------------------------------------------------------===//
// StridedSlice op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: simple_strided_slice
func.func @simple_strided_slice(%input: tensor<4x8xf32>) -> tensor<3x2xf32> {
  // CHECK: %0 = mhlo.constant dense<[0, 1]> : tensor<2xi32>
  // CHECK-NEXT: %1 = mhlo.constant dense<[3, 7]> : tensor<2xi32>
  // CHECK-NEXT: %2 = mhlo.constant dense<[1, 3]> : tensor<2xi32>
  // CHECK-NEXT: %3 = "mhlo.slice"(%arg0)
  %begin = "tf.Const"() {value = dense<[0, 1]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %end = "tf.Const"() {value = dense<[3, 7]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %strides = "tf.Const"() {value = dense<[1, 3]> : tensor<2xi32>} : () -> (tensor<2xi32>)

  %output = "tf.StridedSlice"(%input, %begin, %end, %strides)
      : (tensor<4x8xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<3x2xf32>
  func.return %output : tensor<3x2xf32>
  // CHECK: return %4 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: strided_slice_with_uknown_begin
func.func @strided_slice_with_uknown_begin(%input: tensor<4x8xf32>, %begin: tensor<1xi32>) -> tensor<4x8xf32> {
  %end = "tf.Const"() {value = dense<[3]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  %strides = "tf.Const"() {value = dense<[1]> : tensor<1xi32>} : () -> (tensor<1xi32>)

  // CHECK: %[[DYNAMIC_SLICE:.*]] = "mhlo.dynamic_slice"
  %output = "tf.StridedSlice"(%input, %begin, %end, %strides)
      : (tensor<4x8xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4x8xf32>
  func.return %output : tensor<4x8xf32>
}

//===----------------------------------------------------------------------===//
// Fused op legalizations.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: fused_conv2d
func.func @fused_conv2d(%input: tensor<1x300x300x40xi8>,
                        %filter: tensor<3x3x40x40xi8>,
                        %bias: tensor<40xf32>,
                        %act: tensor<0xi8>) -> tensor<1x300x300x40xi8> {

  // CHECK:       %[[v0:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-NEXT:  %[[v1:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
  // CHECK-NEXT:  %[[v2:.*]] = mhlo.constant dense<2.000000e+00> : tensor<f32>
  // CHECK-NEXT:  %[[v3:.*]] = mhlo.constant dense<-1.280000e+02> : tensor<f32>
  // CHECK-NEXT:  %[[v4:.*]] = "mhlo.broadcast_in_dim"(%3) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<1x300x300x40xf32>
  // CHECK-NEXT:  %[[v5:.*]] = mhlo.convert %arg0 : (tensor<1x300x300x40xi8>) -> tensor<1x300x300x40xf32>
  // CHECK-NEXT:  %[[v6:.*]] = mhlo.convert %arg1 : (tensor<3x3x40x40xi8>) -> tensor<3x3x40x40xf32>
  // CHECK:       %[[v7:.*]] = mhlo.convolution(%[[v5]], %[[v6]])
  // CHECK-SAME{LITERAL}:  dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
  // CHECK-SAME{LITERAL}:  window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]}
  // CHECK-SAME:  batch_group_count = 1
  // CHECK-SAME:  feature_group_count = 1
  // CHECK-NEXT:  %[[v8:.*]] = mhlo.convert %7 : tensor<1x300x300x40xf32>
  // CHECK-NEXT:  %[[v9:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-NEXT:  %[[v10:.*]] = "mhlo.broadcast_in_dim"(%9) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<1x300x300x40xf32>
  // CHECK-NEXT:  %11 = mhlo.multiply %8, %10 : tensor<1x300x300x40xf32>
  // CHECK-NEXT:  %12 = mhlo.convert %arg2 : tensor<40xf32>
  // CHECK-NEXT:  %13 = "mhlo.broadcast_in_dim"(%12) <{broadcast_dimensions = dense<3> : tensor<1xi64>}> : (tensor<40xf32>) -> tensor<1x300x300x40xf32>
  // CHECK-NEXT:  %14 = mhlo.add %11, %13 : tensor<1x300x300x40xf32>
  // CHECK-NEXT:  %15 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT:  %16 = "mhlo.broadcast_in_dim"(%15) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<1x300x300x40xf32>
  // CHECK-NEXT:  %17 = mhlo.maximum %14, %16 : tensor<1x300x300x40xf32>
  // CHECK-NEXT:  %18 = mhlo.constant dense<1.270000e+02> : tensor<f32>
  // CHECK-NEXT:  %19 = "mhlo.broadcast_in_dim"(%18) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<1x300x300x40xf32>
  // CHECK-NEXT:  %20 = mhlo.clamp %4, %17, %19 : tensor<1x300x300x40xf32>
  // CHECK-NEXT:  %21 = mhlo.round_nearest_even %20 : tensor<1x300x300x40xf32>
  // CHECK-NEXT:  %22 = mhlo.convert %21 : (tensor<1x300x300x40xf32>) -> tensor<1x300x300x40xi8>
  // CHECK-NEXT:  return %22 : tensor<1x300x300x40xi8>
  %input_scale = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  %side_input_scale = "tf.Const"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
  %conv2d = "tf._FusedConv2D"(%input, %filter, %bias, %act, %input_scale, %side_input_scale) {
    data_format = "NHWC", dilations = [1, 1, 1, 1], epsilon = 9.99999974E-5 : f32, explicit_paddings = [], filter_format = "HWIO", fused_ops = ["BiasAdd", "Relu"], leakyrelu_alpha = 2.000000e-01 : f32, num_args = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 2, 2>, padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true
    } : (tensor<1x300x300x40xi8>, tensor<3x3x40x40xi8>, tensor<40xf32>, tensor<0xi8>, tensor<f32>, tensor<f32>) -> tensor<1x300x300x40xi8>
  func.return %conv2d : tensor<1x300x300x40xi8>
}

//===----------------------------------------------------------------------===//
// tf.UniqueV2 legalization
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @doesnt_legalize_uniquev2
func.func @doesnt_legalize_uniquev2(%arg0: tensor<3xf32>) -> (tensor<?xf32>, tensor<3xi32>) {
  //CHECK:        %0 = mhlo.constant dense<0> : tensor<1xi32>
  //CHECK-NEXT:   %y, %idx = "tf.UniqueV2"(%arg0, %0) {device = ""} : (tensor<3xf32>, tensor<1xi32>) -> (tensor<?xf32>, tensor<3xi32>)
  //CHECK-NEXT:   return %y, %idx : tensor<?xf32>, tensor<3xi32>

  %cst = "tf.Const"() {device = "", value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %y, %idx = "tf.UniqueV2"(%arg0, %cst) {device = ""} : (tensor<3xf32>, tensor<1xi32>) -> (tensor<?xf32>, tensor<3xi32>)
  func.return %y, %idx : tensor<?xf32>, tensor<3xi32>
}

}
