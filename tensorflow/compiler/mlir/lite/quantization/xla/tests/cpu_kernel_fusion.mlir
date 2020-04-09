// RUN: tf-opt -xla-hlo-cpu-fusion %s | FileCheck %s

// CHECK-LABEL: @mul_add_source
func @mul_add_source(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>) {
  %0 = "xla_hlo.multiply"(%arg0, %arg1) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "xla_hlo.add"(%0, %arg2) {broadcast_dimensions = dense<[]> : tensor<0xi64>} :  (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>

// CHECK: %[[region:.*]] = "quant.region"(%arg0, %arg1, %arg2) ( {
// CHECK: ^bb0(%arg3: tensor<4xf32>, %arg4: tensor<4xf32>, %arg5: tensor<4xf32>):	// no predecessors
// CHECK:   %[[mul:.*]] = xla_hlo.multiply %arg3, %arg4 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : tensor<4xf32>
// CHECK:   %[[add:.*]] = xla_hlo.add %[[mul]], %arg5 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : tensor<4xf32>
// CHECK:   "quant.return"(%[[add]]) : (tensor<4xf32>) -> ()
// CHECK: }) {input_specs = [f32, f32, f32], logical_kernel = "generic.mul_add", output_specs = [f32]} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK: return %[[region]] : tensor<4xf32>
}

// CHECK-LABEL: @mul_add_annotated
func @mul_add_annotated(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>, %arg2: tensor<2x4xf32>) -> (tensor<2x4xf32>) {
  %cst = constant dense<0.0> : tensor<f32>
  %cst_0 = constant dense<255.0> : tensor<f32>
  %cst_1 = constant dense<8> : tensor<i32>
  %cst_2 = constant dense<false> : tensor<i1>
  %qin = "xla_hlo.custom_call"(%arg0, %cst, %cst_0, %cst_1, %cst_2) {backend_config = "", call_target_name = "fake_quant_with_min_max_vars",
    has_side_effect = false, name = "custom-call.1"} : (tensor<2x4xf32>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<i1>) -> tensor<2x4xf32>
  %qw = "xla_hlo.custom_call"(%arg1, %cst, %cst_0, %cst_1, %cst_2) {backend_config = "", call_target_name = "fake_quant_with_min_max_vars",
    has_side_effect = false, name = "custom-call.2"} : (tensor<2x4xf32>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<i1>) -> tensor<2x4xf32>
  %0 = "xla_hlo.multiply"(%qin, %qw) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %1 = "xla_hlo.add"(%0, %arg2) {broadcast_dimensions = dense<[]> : tensor<0xi64>} :  (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %r = "xla_hlo.custom_call"(%1, %cst, %cst_0, %cst_1, %cst_2) {backend_config = "", call_target_name = "fake_quant_with_min_max_vars",
    has_side_effect = false, name = "custom-call.3"} : (tensor<2x4xf32>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<i1>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>

// CHECK: %[[region:.*]] = "quant.region"
// CHECK: ^bb0(%arg3: tensor<2x4xf32>, %arg4: tensor<2x4xf32>, %arg5: tensor<2x4xf32>):	// no predecessors
// CHECK:   %[[mul:.*]] = xla_hlo.multiply %arg3, %arg4 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : tensor<2x4xf32>
// CHECK:   %[[add:.*]] = xla_hlo.add %[[mul]], %arg5 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : tensor<2x4xf32>
// CHECK:   "quant.return"(%[[add]]) : (tensor<2x4xf32>) -> ()
// CHECK: }) {input_specs = [!quant.uniform<i8:f32, 1.000000e+00:-128>, !quant.uniform<i8:f32, 1.000000e+00:-128>, f32],
// CHECK-SAME: logical_kernel = "generic.mul_add", output_specs = [!quant.uniform<i8:f32, 1.000000e+00:-128>]} :
// CHECK-SAME: (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:  %[[r:.*]] = "xla_hlo.custom_call"(%[[region]]
// CHECK: return %[[r]] : tensor<2x4xf32>
}

// CHECK-LABEL: @reduce_window
func @reduce_window(%arg0: tensor<1x28x28x32xf32>, %arg1: tensor<f32>) -> (tensor<1x14x14x32xf32>) {
  %0 = "xla_hlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = xla_hlo.maximum %arg2, %arg3 : tensor<f32>
    "xla_hlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    base_dilations = dense<1> : tensor<4xi64>,
    padding = dense<0> : tensor<4x2xi64>,
    window_dilations = dense<1> : tensor<4xi64>,
    window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
  } : (tensor<1x28x28x32xf32>, tensor<f32>) -> tensor<1x14x14x32xf32>
  return %0 : tensor<1x14x14x32xf32>

// CHECK: "quant.region"(%arg0, %arg1) ( {
// CHECK: ^bb0(%arg2: tensor<1x28x28x32xf32>, %arg3: tensor<f32>):	// no predecessors
// CHECK:   %[[rw:.*]] = "xla_hlo.reduce_window"(%arg2, %arg3) ( {
// CHECK:   ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):	// no predecessors
// CHECK:     %[[max:.*]] = xla_hlo.maximum %arg4, %arg5 : tensor<f32>
// CHECK:     "xla_hlo.return"(%[[max]]) : (tensor<f32>) -> ()
// CHECK:   })
// CHECK:   "quant.return"(%[[rw]])
// CHECK: }) {input_specs = [f32, f32], logical_kernel = "generic.reduce_window", output_specs = [f32]}
}

// CHECK-LABEL: @reshape
func @reshape(%arg0: tensor<1x7x7x64xf32>) -> (tensor<1x3136xf32>) {
  %0 = "xla_hlo.reshape"(%arg0) : (tensor<1x7x7x64xf32>) -> tensor<1x3136xf32>
  return %0 : tensor<1x3136xf32>

// CHECK: "quant.region"(%arg0)
// CHECK: logical_kernel = "generic.reshape"
}

// CHECK-LABEL: @broadcast
func @broadcast(%arg0: tensor<64xf32>) -> (tensor<1x14x14x64xf32>) {
  %0 = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<1x14x14x64xf32>
  return %0 : tensor<1x14x14x64xf32>

// CHECK: "quant.region"(%arg0)
// CHECK: logical_kernel = "generic.broadcast"
}

// CHECK-LABEL: @biased_dot
func @biased_dot(%arg0: tensor<1x1024xf32>, %arg1: tensor<1024x10xf32>, %arg2: tensor<1x10xf32>) -> (tensor<1x10xf32>) {
  %0 = "xla_hlo.dot"(%arg0, %arg1) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x1024xf32>, tensor<1024x10xf32>) -> tensor<1x10xf32>
  %1 = xla_hlo.add %0, %arg2 : tensor<1x10xf32>
  return %1 : tensor<1x10xf32>

// CHECK: "quant.region"(%arg0, %arg1, %arg2)
// CHECK: xla_hlo.dot
// CHECK: xla_hlo.add
// CHECK: logical_kernel = "generic.biased_dot"
}

// CHECK-LABEL: @biased_conv
func @biased_conv(%arg0: tensor<1x14x14x32xf32>, %arg1: tensor<5x5x32x64xf32>, %arg2: tensor<1x14x14x64xf32>) -> (tensor<1x14x14x64xf32>) {
  %0 = "xla_hlo.convolution"(%arg0, %arg1) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64,
    input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64,
    kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64,
    output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, lhs_dilations = dense<1> : tensor<2xi64>,
    padding = dense<2> : tensor<2x2xi64>, precision_config = ["DEFAULT", "DEFAULT"], rhs_dilations = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>
  } : (tensor<1x14x14x32xf32>, tensor<5x5x32x64xf32>) -> tensor<1x14x14x64xf32>
  %1 = xla_hlo.add %0, %arg2 : tensor<1x14x14x64xf32>
  return %1 : tensor<1x14x14x64xf32>

// CHECK: "quant.region"(%arg0, %arg1, %arg2)
// CHECK: xla_hlo.convolution
// CHECK: xla_hlo.add
// CHECK: logical_kernel = "generic.biased_conv"
}

// CHECK-LABEL: @biased_dot_relu
func @biased_dot_relu(%arg0: tensor<1x1024xf32>, %arg1: tensor<1024x10xf32>, %arg2: tensor<1x10xf32>) -> (tensor<1x10xf32>) {
  %cst = constant dense<0.0> : tensor<1x10xf32>
  %0 = "xla_hlo.dot"(%arg0, %arg1) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x1024xf32>, tensor<1024x10xf32>) -> tensor<1x10xf32>
  %1 = xla_hlo.add %0, %arg2 : tensor<1x10xf32>
  %2 = xla_hlo.maximum %1, %cst : tensor<1x10xf32>
  return %2 : tensor<1x10xf32>

// CHECK: "quant.region"(%arg0, %arg1, %arg2)
// CHECK: constant
// CHECK: xla_hlo.dot
// CHECK: xla_hlo.add
// CHECK: xla_hlo.maximum
// CHECK: logical_kernel = "generic.biased_dot_relu"
}

// CHECK-LABEL: @biased_conv_relu
func @biased_conv_relu(%arg0: tensor<1x14x14x32xf32>, %arg1: tensor<5x5x32x64xf32>, %arg2: tensor<1x14x14x64xf32>) -> (tensor<1x14x14x64xf32>) {
  %cst = constant dense<0.0> : tensor<1x14x14x64xf32>
  %0 = "xla_hlo.convolution"(%arg0, %arg1) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64,
    input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64,
    kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64,
    output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, lhs_dilations = dense<1> : tensor<2xi64>,
    padding = dense<2> : tensor<2x2xi64>, precision_config = ["DEFAULT", "DEFAULT"], rhs_dilations = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>
  } : (tensor<1x14x14x32xf32>, tensor<5x5x32x64xf32>) -> tensor<1x14x14x64xf32>
  %1 = xla_hlo.add %0, %arg2 : tensor<1x14x14x64xf32>
  %2 = xla_hlo.maximum %1, %cst : tensor<1x14x14x64xf32>
  return %2 : tensor<1x14x14x64xf32>

// CHECK: "quant.region"(%arg0, %arg1, %arg2)
// CHECK: constant
// CHECK: xla_hlo.convolution
// CHECK: xla_hlo.add
// CHECK: xla_hlo.maximum
// CHECK: logical_kernel = "generic.biased_conv_relu"
}

// CHECK-LABEL: @biased_conv_relu_shared
func @biased_conv_relu_shared(%arg0: tensor<1x14x14x32xf32>, %arg1: tensor<5x5x32x64xf32>, %arg2: tensor<1x14x14x64xf32>) -> (tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) {
  %cst = constant dense<0.0> : tensor<1x14x14x64xf32>
  %0 = "xla_hlo.convolution"(%arg0, %arg1) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64,
    input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64,
    kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64,
    output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, lhs_dilations = dense<1> : tensor<2xi64>,
    padding = dense<2> : tensor<2x2xi64>, precision_config = ["DEFAULT", "DEFAULT"], rhs_dilations = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>
  } : (tensor<1x14x14x32xf32>, tensor<5x5x32x64xf32>) -> tensor<1x14x14x64xf32>
  %1 = xla_hlo.add %0, %arg2 : tensor<1x14x14x64xf32>
  %2 = xla_hlo.maximum %1, %cst : tensor<1x14x14x64xf32>
  return %cst, %2 : tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>

// CHECK: "quant.region"(%arg0, %arg1, %arg2)
// CHECK: constant
// CHECK: xla_hlo.convolution
// CHECK: xla_hlo.add
// CHECK: %[[max:.*]] = xla_hlo.maximum
// CHECK: "quant.return"(%[[max]])
// CHECK: logical_kernel = "generic.biased_conv_relu"
}

// CHECK-LABEL: @biased_conv_relu6
func @biased_conv_relu6(%arg0: tensor<1x14x14x32xf32>, %arg1: tensor<5x5x32x64xf32>, %arg2: tensor<1x14x14x64xf32>) -> (tensor<1x14x14x64xf32>) {
  %min = constant dense<0.0> : tensor<1x14x14x64xf32>
  %max = constant dense<6.0> : tensor<1x14x14x64xf32>
  %0 = "xla_hlo.convolution"(%arg0, %arg1) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64,
    input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64,
    kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64,
    output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, lhs_dilations = dense<1> : tensor<2xi64>,
    padding = dense<2> : tensor<2x2xi64>, precision_config = ["DEFAULT", "DEFAULT"], rhs_dilations = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>
  } : (tensor<1x14x14x32xf32>, tensor<5x5x32x64xf32>) -> tensor<1x14x14x64xf32>
  %1 = xla_hlo.add %0, %arg2 : tensor<1x14x14x64xf32>
  %2 = "xla_hlo.clamp"(%min, %1, %max) : (tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>, tensor<1x14x14x64xf32>) -> tensor<1x14x14x64xf32>
  return %2 : tensor<1x14x14x64xf32>

// CHECK: "quant.region"(%arg0, %arg1, %arg2)
// CHECK: constant
// CHECK: constant
// CHECK: xla_hlo.convolution
// CHECK: xla_hlo.add
// CHECK: xla_hlo.clamp
// CHECK: logical_kernel = "generic.biased_conv_relu6"
}
