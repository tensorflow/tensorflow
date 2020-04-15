// RUN: xla-opt %s -pass-pipeline='func(canonicalize)' | FileCheck %s --dump-input-on-failure

func @dynamic_slice_variable_start(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  // CHECK: "xla_hlo.dynamic-slice"
  %1 = "xla_hlo.dynamic-slice"(%arg0, %arg1, %arg2) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  return %1 : tensor<1x4xi32>
}

// CHECK-LABEL: dynamic_slice_constant_start
func @dynamic_slice_constant_start(%arg0: tensor<4xi32>) -> tensor<2xi32> {
  // CHECK: %[[RESULT:.*]] =  "xla_hlo.slice"(%arg0)
  // CHECK-DAG-SAME: limit_indices = dense<3> : tensor<1xi64>
  // CHECK-DAG-SAME: start_indices = dense<1> : tensor<1xi64>
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>}
  // CHECK: return %[[RESULT]] : tensor<2xi32>
  %0 = xla_hlo.constant dense<1> : tensor<i64>
  %1 = "xla_hlo.dynamic-slice"(%arg0, %0) {slice_sizes = dense<2> : tensor<1xi64>} : (tensor<4xi32>, tensor<i64>) -> tensor<2xi32>
  return %1 : tensor<2xi32>
}

// CHECK-LABEL: dynamic_slice_constant_start_dynamic_shape
func @dynamic_slice_constant_start_dynamic_shape(%arg0: tensor<?x4xi32>, %arg1: tensor<2xi64>) -> tensor<?x4xi32> {
  // CHECK: %[[RESULT:.*]] = "xla_hlo.slice"(%arg0)
  // CHECK-DAG-SAME: limit_indices = dense<[2, 4]> : tensor<2xi64>
  // CHECK-DAG-SAME: start_indices = dense<[1, 0]> : tensor<2xi64>
  // CHECK-DAG-SAME: strides = dense<1> : tensor<2xi64>
  // CHECK: return %[[RESULT]] : tensor<?x4xi32>
  %0 = xla_hlo.constant dense<1> : tensor<i64>
  %1 = xla_hlo.constant dense<0> : tensor<i64>
  %2 = "xla_hlo.dynamic-slice"(%arg0, %0, %1) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<?x4xi32>, tensor<i64>, tensor<i64>) -> tensor<?x4xi32>
  return %2 : tensor<?x4xi32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic
func @dynamic_broadcast_in_dim_op_not_actually_dynamic(%arg0: tensor<4xf32>, %arg1: tensor<2xi64>) -> tensor<5x4xf32> {
  // CHECK: %[[RESULT:.+]] = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<5x4xf32>
  %0 = "xla_hlo.dynamic_broadcast_in_dim"(%arg0, %arg1) { broadcast_dimensions = dense<1> : tensor<1xi64> } : (tensor<4xf32>, tensor<2xi64>) -> tensor<5x4xf32>
  // CHECK: return %[[RESULT]] : tensor<5x4xf32>
  return %0 : tensor<5x4xf32>
}

// CHECK-LABEL: @complex_expand_fold
func @complex_expand_fold(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = "xla_hlo.complex"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xcomplex<f32>>)
  %1 = "xla_hlo.real"(%0) : (tensor<4xcomplex<f32>>) -> (tensor<4xf32>)
  %2 = "xla_hlo.imag"(%0) : (tensor<4xcomplex<f32>>) -> (tensor<4xf32>)
  // CHECK: return %arg0, %arg1
  return %1, %2 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: @complex_collapse_fold
func @complex_collapse_fold(%arg0: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %0 = "xla_hlo.real"(%arg0) : (tensor<4xcomplex<f32>>) -> (tensor<4xf32>)
  %1 = "xla_hlo.imag"(%arg0) : (tensor<4xcomplex<f32>>) -> (tensor<4xf32>)
  %2 = "xla_hlo.complex"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xcomplex<f32>>
  // CHECK: return %arg0
  return %2 : tensor<4xcomplex<f32>>
}

// CHECK-LABEL: @iota_not_lowered_to_constant
func @iota_not_lowered_to_constant() -> tensor<4xi32> {
  // CHECK: [[RESULT:%.*]] = "xla_hlo.iota"
  // CHECK: return [[RESULT]]
  %0 = "xla_hlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: @unary_einsum
func @unary_einsum(%arg0: tensor<2x3xf32>) -> tensor<2x2xf32> {
  // CHECK: %[[ONE:.*]] = xla_hlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: "xla_hlo.einsum"(%[[ONE]], %arg0) {einsum_config = ",ab->aa"}
  %0 = "xla_hlo.unary_einsum"(%arg0) {einsum_config = "ab->aa"} : (tensor<2x3xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: @extract_scalars_to_tensor
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32
func @extract_scalars_to_tensor(%arg0: i32, %arg1: i32) -> i32 {
  %0 = "xla_hlo.scalars_to_dimension_tensor"(%arg0, %arg1) : (i32, i32) -> tensor<2xi32>
  %1 = constant 0 : index
  %2 = extract_element %0[%1] : tensor<2xi32>
  // CHECK: return %[[ARG0]]
  return %2 : i32
}

// CHECK-LABEL: func @fold_copy
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @fold_copy(%arg : tensor<1x4xf32>) -> tensor<1x4xf32> {
  // CHECK: return [[ARG]]
  %0 = "xla_hlo.copy"(%arg) : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// CHECK-LABEL: func @fold_pad_into_conv_f32
func @fold_pad_into_conv_f32(%arg0 : tensor<1x32x32x3xf32>,
                         %arg1 : tensor<7x7x3x64xf32>)
    -> tensor<1x16x16x64xf32> {
  //  CHECK-NOT: xla_hlo.pad
  //      CHECK: xla_hlo.convolution
  // CHECK-SAME: padding = dense<3> : tensor<2x2xi64>
  %0 = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "xla_hlo.pad"(%arg0, %0) {
    edge_padding_high = dense<[0, 3, 3, 0]> : tensor<4xi64>,
    edge_padding_low = dense<[0, 3, 3, 0]> : tensor<4xi64>,
    interior_padding = dense<0> : tensor<4xi64>
  } : (tensor<1x32x32x3xf32>, tensor<f32>) -> tensor<1x38x38x3xf32>
  %2 = "xla_hlo.convolution"(%1, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = {
      input_batch_dimension = 0 : i64,
      input_feature_dimension = 3 : i64,
      input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
      kernel_input_feature_dimension = 2 : i64,
      kernel_output_feature_dimension = 3 : i64,
      kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
      output_batch_dimension = 0 : i64,
      output_feature_dimension = 3 : i64,
      output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>
    },
    feature_group_count = 1 : i64,
    padding = dense<0> : tensor<2x2xi64>,
    window_strides = dense<2> : tensor<2xi64>
  } : (tensor<1x38x38x3xf32>, tensor<7x7x3x64xf32>) -> tensor<1x16x16x64xf32>
  return %2 : tensor<1x16x16x64xf32>
}

// CHECK-LABEL: func @fold_pad_into_conv_i32
func @fold_pad_into_conv_i32(%arg0 : tensor<1x32x32x3xi32>,
                         %arg1 : tensor<7x7x3x64xi32>)
    -> tensor<1x16x16x64xi32> {
  //  CHECK-NOT: xla_hlo.pad
  //      CHECK: xla_hlo.convolution
  // CHECK-SAME: padding = dense<3> : tensor<2x2xi64>
  %0 = xla_hlo.constant dense<0> : tensor<i32>
  %1 = "xla_hlo.pad"(%arg0, %0) {
    edge_padding_high = dense<[0, 3, 3, 0]> : tensor<4xi64>,
    edge_padding_low = dense<[0, 3, 3, 0]> : tensor<4xi64>,
    interior_padding = dense<0> : tensor<4xi64>
  } : (tensor<1x32x32x3xi32>, tensor<i32>) -> tensor<1x38x38x3xi32>
  %2 = "xla_hlo.convolution"(%1, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = {
      input_batch_dimension = 0 : i64,
      input_feature_dimension = 3 : i64,
      input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
      kernel_input_feature_dimension = 2 : i64,
      kernel_output_feature_dimension = 3 : i64,
      kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
      output_batch_dimension = 0 : i64,
      output_feature_dimension = 3 : i64,
      output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>
    },
    feature_group_count = 1 : i64,
    window_strides = dense<2> : tensor<2xi64>
  } : (tensor<1x38x38x3xi32>, tensor<7x7x3x64xi32>) -> tensor<1x16x16x64xi32>
  return %2 : tensor<1x16x16x64xi32>
}

// CHECK-LABEL: func @dynamic_reshape_not_actually_dynamic
func @dynamic_reshape_not_actually_dynamic(%arg0: tensor<4xf32>, %shape: tensor<2xindex>) -> tensor<4x1xf32> {
  // CHECK: xla_hlo.reshape
  %0 = "xla_hlo.dynamic_reshape"(%arg0, %shape) : (tensor<4xf32>, tensor<2xindex>) -> tensor<4x1xf32>
  return %0 : tensor<4x1xf32>
}

// CHECK-LABEL: do_not_dce_while
func @do_not_dce_while(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK: xla_hlo.while
  %0 = "xla_hlo.while"(%arg0) ( {
  ^bb0(%arg1: tensor<i64>):
    %1 = "xla_hlo.compare"(%arg1, %arg1) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    "xla_hlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i64>):
    %1 = "xla_hlo.create_token"() : () -> !xla_hlo.token
    // Side-effecting op outfeed present inside while.
    %2 = "xla_hlo.outfeed"(%arg1, %1) {outfeed_config = ""} : (tensor<i64>, !xla_hlo.token) -> !xla_hlo.token
    "xla_hlo.return"(%arg1) : (tensor<i64>) -> ()
  }) : (tensor<i64>) -> tensor<i64>

  return %arg0 : tensor<i64>
}
