// RUN: xla-opt %s -pass-pipeline='func(canonicalize)' | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: add_fold
func @add_fold() -> tensor<4xi64> {
  %0 = xla_hlo.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
  %1 = xla_hlo.constant dense<[5, 6, 7, 8]> : tensor<4xi64>
  // CHECK: xla_hlo.constant dense<[6, 8, 10, 12]>
  %2 = "xla_hlo.add"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  return %2 : tensor<4xi64>
}

// CHECK-LABEL: add_scalar_fold
func @add_scalar_fold() -> tensor<4xi64> {
  %0 = xla_hlo.constant dense<1> : tensor<4xi64>
  %1 = xla_hlo.constant dense<5> : tensor<4xi64>
  // CHECK: xla_hlo.constant dense<6>
  %2 = "xla_hlo.add"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  return %2 : tensor<4xi64>
}

// CHECK-LABEL: add_fold_float
func @add_fold_float() -> tensor<4xf64> {
  %0 = xla_hlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf64>
  %1 = xla_hlo.constant dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf64>
  // CHECK: xla_hlo.constant dense<[6.000000e+00, 8.000000e+00, 1.000000e+01, 1.200000e+01]>
  %2 = "xla_hlo.add"(%0, %1) : (tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>)
  return %2 : tensor<4xf64>
}

// CHECK-LABEL: sub_scalar_fold
func @sub_scalar_fold() -> tensor<4xi64> {
  %0 = xla_hlo.constant dense<5> : tensor<4xi64>
  %1 = xla_hlo.constant dense<1> : tensor<4xi64>
  // CHECK: xla_hlo.constant dense<4>
  %2 = "xla_hlo.subtract"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  return %2 : tensor<4xi64>
}

// CHECK-LABEL: multiply_scalar_fold
func @multiply_scalar_fold() -> tensor<4xi64> {
  %0 = xla_hlo.constant dense<5> : tensor<4xi64>
  %1 = xla_hlo.constant dense<3> : tensor<4xi64>
  // CHECK: xla_hlo.constant dense<15>
  %2 = "xla_hlo.multiply"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  return %2 : tensor<4xi64>
}

// CHECK-LABEL: divide_scalar_fold
func @divide_scalar_fold() -> tensor<4xi64> {
  %0 = xla_hlo.constant dense<7> : tensor<4xi64>
  %1 = xla_hlo.constant dense<5> : tensor<4xi64>
  // CHECK: xla_hlo.constant dense<1>
  %2 = "xla_hlo.divide"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  return %2 : tensor<4xi64>
}

// CHECK-LABEL: divide_fold_float
func @divide_fold_float() -> tensor<4xf64> {
  %0 = xla_hlo.constant dense<[5.0, 66.0, 5.0, 1.0]> : tensor<4xf64>
  %1 = xla_hlo.constant dense<[5.0, 3.0, 2.0, 4.0]> : tensor<4xf64>
  // CHECK: xla_hlo.constant dense<[1.000000e+00, 2.200000e+01, 2.500000e+00, 2.500000e-01]>
  %2 = "xla_hlo.divide"(%0, %1) : (tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>)
  return %2 : tensor<4xf64>
}

// CHECK-LABEL: max_scalar_fold
func @max_scalar_fold() -> tensor<4xi64> {
  %0 = xla_hlo.constant dense<7> : tensor<4xi64>
  %1 = xla_hlo.constant dense<5> : tensor<4xi64>
  // CHECK: xla_hlo.constant dense<7>
  %2 = "xla_hlo.maximum"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  return %2 : tensor<4xi64>
}

// CHECK-LABEL: max_fold_float
func @max_fold_float() -> tensor<4xf64> {
  %0 = xla_hlo.constant dense<[5.0, 66.0, 5.0, 1.0]> : tensor<4xf64>
  %1 = xla_hlo.constant dense<[5.0, 3.0, 2.0, 4.0]> : tensor<4xf64>
  // CHECK: xla_hlo.constant dense<[5.000000e+00, 6.600000e+01, 5.000000e+00, 4.000000e+00]>
  %2 = "xla_hlo.maximum"(%0, %1) : (tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>)
  return %2 : tensor<4xf64>
}

// CHECK-LABEL: min_scalar_fold
func @min_scalar_fold() -> tensor<4xi64> {
  %0 = xla_hlo.constant dense<7> : tensor<4xi64>
  %1 = xla_hlo.constant dense<-5> : tensor<4xi64>
  // CHECK: xla_hlo.constant dense<-5>
  %2 = "xla_hlo.minimum"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  return %2 : tensor<4xi64>
}

// CHECK-LABEL: min_fold_float
func @min_fold_float() -> tensor<4xf64> {
  %0 = xla_hlo.constant dense<[5.0, 66.0, 5.0, 1.0]> : tensor<4xf64>
  %1 = xla_hlo.constant dense<[5.0, 3.0, 2.0, 4.0]> : tensor<4xf64>
  // CHECK: xla_hlo.constant dense<[5.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00]>
  %2 = "xla_hlo.minimum"(%0, %1) : (tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>)
  return %2 : tensor<4xf64>
}

// CHECK-LABEL: concatenate_noop
func @concatenate_noop(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK-SAME: [[ARG:%.+]]: tensor<4xi32>
  %0 = "xla_hlo.concatenate"(%arg0) { dimension = 0 : i64 } : (tensor<4xi32>) -> tensor<4xi32>

  // CHECK: return [[ARG]]
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: concatenate_remove_operand
func @concatenate_remove_operand(%arg0: tensor<4xi32>, %arg1: tensor<0xi32>) -> tensor<4xi32> {
  // CHECK-SAME: [[ARG0:%.+]]: tensor<4xi32>
  // CHECK-SAME: [[ARG1:%.+]]: tensor<0xi32>
  %0 = "xla_hlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<4xi32>, tensor<0xi32>) -> tensor<4xi32>

  // CHECK: return [[ARG0]]
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: concatenate_empty_bool
func @concatenate_empty_bool(%arg0: tensor<0xi1>, %arg1: tensor<0xi1>) -> tensor<0xi1> {
  // CHECK: xla_hlo.constant
  %0 = "xla_hlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<0xi1>, tensor<0xi1>) -> tensor<0xi1>

  return %0 : tensor<0xi1>
}

// CHECK-LABEL: concatenate_empty_int
func @concatenate_empty_int(%arg0: tensor<0xi32>, %arg1: tensor<0xi32>) -> tensor<0xi32> {
  // CHECK: xla_hlo.constant
  %0 = "xla_hlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<0xi32>, tensor<0xi32>) -> tensor<0xi32>

  return %0 : tensor<0xi32>
}

// CHECK-LABEL: concatenate_empty_float
func @concatenate_empty_float(%arg0: tensor<0xf32>, %arg1: tensor<0xf32>) -> tensor<0xf32> {
  // CHECK: xla_hlo.constant
  %0 = "xla_hlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<0xf32>, tensor<0xf32>) -> tensor<0xf32>

  return %0 : tensor<0xf32>
}

// CHECK-LABEL: concatenate_const_1D
func @concatenate_const_1D() -> tensor<4xi32> {
  // CHECK: [[VAL:%.+]]= xla_hlo.constant dense<[0, 1, 2, 3]>
  %0 = xla_hlo.constant dense<[0, 1]> : tensor<2xi32>
  %1 = xla_hlo.constant dense<[2, 3]> : tensor<2xi32>
  %2 = "xla_hlo.concatenate"(%0, %1) { dimension = 0 : i64 } : (tensor<2xi32>, tensor<2xi32>) -> tensor<4xi32>

  // CHECK: return [[VAL]]
  return %2 : tensor<4xi32>
}

// CHECK-LABEL: concatenate_const_1D_float
func @concatenate_const_1D_float() -> tensor<4xf32> {
  // CHECK: [[VAL:%.+]] = xla_hlo.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]>

  %0 = xla_hlo.constant dense<[0.0, 1.0]> : tensor<2xf32>
  %1 = xla_hlo.constant dense<[2.0, 3.0]> : tensor<2xf32>
  %2 = "xla_hlo.concatenate"(%0, %1) { dimension = 0 : i64 } : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>

  // CHECK: return [[VAL]]
  return %2 : tensor<4xf32>
}

// CHECK-LABEL: concatenate_const_2D_vertical
func @concatenate_const_2D_vertical() -> tensor<2x2xi32> {
  // CHECK: [[VAL:%.+]]= xla_hlo.constant dense<[
  // CHECK-SAME: [0, 1], [2, 3]
  // CHECK-SAME: ]>
  %0 = xla_hlo.constant dense<[[0, 1]]> : tensor<1x2xi32>
  %1 = xla_hlo.constant dense<[[2, 3]]> : tensor<1x2xi32>
  %2 = "xla_hlo.concatenate"(%0, %1) { dimension = 0 : i64 } : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>

  // CHECK: return [[VAL]]
  return %2 : tensor<2x2xi32>
}

// CHECK-LABEL: concatenate_const_2D_horizontal
func @concatenate_const_2D_horizontal() -> tensor<2x2xi32> {
  // CHECK: [[VAL:%.+]]= xla_hlo.constant dense<[
  // CHECK-SAME: [0, 2], [1, 3]
  // CHECK-SAME: ]>
  %0 = xla_hlo.constant dense<[[0], [1]]> : tensor<2x1xi32>
  %1 = xla_hlo.constant dense<[[2], [3]]> : tensor<2x1xi32>
  %2 = "xla_hlo.concatenate"(%0, %1) { dimension = 1 : i64 } : (tensor<2x1xi32>, tensor<2x1xi32>) -> tensor<2x2xi32>

  // CHECK: return [[VAL]]
  return %2 : tensor<2x2xi32>
}

// CHECK-LABEL: dynamic_slice_variable_start
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

// CHECK-LABEL: slice_2D_noop
// CHECK-SAME: [[ARG:%.+]]: tensor<2x2xi64>
func @slice_2D_noop(%arg0: tensor<2x2xi64>) -> tensor<2x2xi64> {
  %0 = "xla_hlo.slice"(%arg0) { limit_indices = dense<[2, 2]> : tensor<2xi64>, start_indices = dense<[0, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xi64>) -> (tensor<2x2xi64>)

  // CHECK-NEXT: return [[ARG]]
  return %0 : tensor<2x2xi64>
}

// CHECK-LABEL: slice_1D_fold
func @slice_1D_fold() -> tensor<2xi64> {
  %0 = xla_hlo.constant dense<[5, 7, 9, 10]> : tensor<4xi64>
  // CHECK: xla_hlo.constant dense<[7, 9]>
  %1 = "xla_hlo.slice"(%0) { limit_indices = dense<[3]> : tensor<1xi64>, start_indices = dense<[1]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xi64>) -> (tensor<2xi64>)
  return %1 : tensor<2xi64>
}

// CHECK-LABEL: slice_1D_fp
func @slice_1D_fp() -> tensor<2xf32> {
  %0 = xla_hlo.constant dense<[5.0, 7.0, 9.0, 10.0]> : tensor<4xf32>
  // CHECK: xla_hlo.constant dense<[7.000000e+00, 9.000000e+00]>
  %1 = "xla_hlo.slice"(%0) { limit_indices = dense<[3]> : tensor<1xi64>, start_indices = dense<[1]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> (tensor<2xf32>)
  return %1 : tensor<2xf32>
}

// CHECK-LABEL: slice_1D_strided_fold
func @slice_1D_strided_fold() -> tensor<2xi64> {
  %0 = xla_hlo.constant dense<[5, 7, 9, 10]> : tensor<4xi64>
  // CHECK: xla_hlo.constant dense<[7, 10]>
  %1 = "xla_hlo.slice"(%0) { limit_indices = dense<[4]> : tensor<1xi64>, start_indices = dense<[1]> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>} : (tensor<4xi64>) -> (tensor<2xi64>)
  return %1 : tensor<2xi64>
}

// CHECK-LABEL: slice_2D_fold
func @slice_2D_fold() -> tensor<2x2xi64> {
  %0 = xla_hlo.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi64>
  // CHECK-NEXT: xla_hlo.constant dense<[
  // CHECK-SAME: [6, 7],
  // CHECK-SAME: [10, 11]
  // CHECK-SAME: ]>
  %1 = "xla_hlo.slice"(%0) { limit_indices = dense<[3, 4]> : tensor<2xi64>, start_indices = dense<[1, 2]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x4xi64>) -> (tensor<2x2xi64>)
  return %1 : tensor<2x2xi64>
}

// CHECK-LABEL: slice_2D_fold_horizontal
func @slice_2D_fold_horizontal() -> tensor<1x4xi64> {
  %0 = xla_hlo.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi64>
  // CHECK-NEXT: xla_hlo.constant dense<[
  // CHECK-SAME: [0, 1, 2, 3]
  // CHECK-SAME: ]>
  %1 = "xla_hlo.slice"(%0) { limit_indices = dense<[1, 4]> : tensor<2xi64>, start_indices = dense<[0, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x4xi64>) -> (tensor<1x4xi64>)
  return %1 : tensor<1x4xi64>
}

// CHECK-LABEL: slice_2D_fold_vertical
func @slice_2D_fold_vertical() -> tensor<4x1xi64> {
  %0 = xla_hlo.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi64>
  // CHECK-NEXT: xla_hlo.constant dense<[
  // CHECK-SAME: [2], [6], [10], [14]
  // CHECK-SAME: ]>
  %1 = "xla_hlo.slice"(%0) { limit_indices = dense<[4, 3]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x4xi64>) -> (tensor<4x1xi64>)
  return %1 : tensor<4x1xi64>
}

// CHECK-LABEL: func @broadcast_in_dim_identity
func @broadcast_in_dim_identity(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  // CHECK: return %arg0
  %0 = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func @broadcast_in_dim_not_identity_because_it_actually_broadcasts
func @broadcast_in_dim_not_identity_because_it_actually_broadcasts(%arg0: tensor<1x2xf32>) -> tensor<2x2xf32> {
  // CHECK: xla_hlo.broadcast_in_dim
  %0 = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func @broadcast_in_dim_not_identity_permutation
func @broadcast_in_dim_not_identity_permutation(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: xla_hlo.broadcast_in_dim
  %0 = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 0]> : tensor<2xi64>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
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

// CHECK-LABEL: @dynamic_iota_is_static
func @dynamic_iota_is_static(%arg0 : tensor<1xindex>) -> tensor<4xi32> {
  // CHECK: [[RESULT:%.*]] = "xla_hlo.iota"
  // CHECK: return [[RESULT]]
  %0 = "xla_hlo.dynamic_iota"(%arg0) {iota_dimension = 0 : i64} : (tensor<1xindex>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
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

// CHECK-LABEL: do_not_dce_while_with_outfeed
func @do_not_dce_while_with_outfeed(%arg0: tensor<i64>) -> tensor<i64> {
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

// CHECK-LABEL: dce_while_without_side_effect
func @dce_while_without_side_effect(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK-NOT: xla_hlo.while
  %0 = "xla_hlo.while"(%arg0) ( {
  ^bb0(%arg1: tensor<i64>):
    %1 = "xla_hlo.compare"(%arg1, %arg1) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    "xla_hlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i64>):
    %1 = "xla_hlo.create_token"() : () -> !xla_hlo.token
    "xla_hlo.return"(%arg1) : (tensor<i64>) -> ()
  }) : (tensor<i64>) -> tensor<i64>

  return %arg0 : tensor<i64>
}
