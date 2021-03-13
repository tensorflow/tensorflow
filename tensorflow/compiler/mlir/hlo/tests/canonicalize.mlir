// RUN: mlir-hlo-opt %s -pass-pipeline='func(canonicalize)' | FileCheck %s

// CHECK-LABEL: add_fold
func @add_fold() -> tensor<4xi64> {
  %0 = mhlo.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
  %1 = mhlo.constant dense<[5, 6, 7, 8]> : tensor<4xi64>
  // CHECK: mhlo.constant dense<[6, 8, 10, 12]>
  %2 = "mhlo.add"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  return %2 : tensor<4xi64>
}

// CHECK-LABEL: add_scalar_fold
func @add_scalar_fold() -> tensor<4xi64> {
  %0 = mhlo.constant dense<1> : tensor<4xi64>
  %1 = mhlo.constant dense<5> : tensor<4xi64>
  // CHECK: mhlo.constant dense<6>
  %2 = "mhlo.add"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  return %2 : tensor<4xi64>
}

// CHECK-LABEL: add_fold_float
func @add_fold_float() -> tensor<4xf64> {
  %0 = mhlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf64>
  %1 = mhlo.constant dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf64>
  // CHECK: mhlo.constant dense<[6.000000e+00, 8.000000e+00, 1.000000e+01, 1.200000e+01]>
  %2 = "mhlo.add"(%0, %1) : (tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>)
  return %2 : tensor<4xf64>
}

// CHECK-LABEL: sub_scalar_fold
func @sub_scalar_fold() -> tensor<4xi64> {
  %0 = mhlo.constant dense<5> : tensor<4xi64>
  %1 = mhlo.constant dense<1> : tensor<4xi64>
  // CHECK: mhlo.constant dense<4>
  %2 = "mhlo.subtract"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  return %2 : tensor<4xi64>
}

// CHECK-LABEL: multiply_scalar_fold
func @multiply_scalar_fold() -> tensor<4xi64> {
  %0 = mhlo.constant dense<5> : tensor<4xi64>
  %1 = mhlo.constant dense<3> : tensor<4xi64>
  // CHECK: mhlo.constant dense<15>
  %2 = "mhlo.multiply"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  return %2 : tensor<4xi64>
}

// CHECK-LABEL: divide_scalar_fold
func @divide_scalar_fold() -> tensor<4xi64> {
  %0 = mhlo.constant dense<7> : tensor<4xi64>
  %1 = mhlo.constant dense<5> : tensor<4xi64>
  // CHECK: mhlo.constant dense<1>
  %2 = "mhlo.divide"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  return %2 : tensor<4xi64>
}

// CHECK-LABEL: divide_fold_float
func @divide_fold_float() -> tensor<4xf64> {
  %0 = mhlo.constant dense<[5.0, 66.0, 5.0, 1.0]> : tensor<4xf64>
  %1 = mhlo.constant dense<[5.0, 3.0, 2.0, 4.0]> : tensor<4xf64>
  // CHECK: mhlo.constant dense<[1.000000e+00, 2.200000e+01, 2.500000e+00, 2.500000e-01]>
  %2 = "mhlo.divide"(%0, %1) : (tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>)
  return %2 : tensor<4xf64>
}

// CHECK-LABEL: remainder_fold_int
func @remainder_fold_int() -> tensor<4xi32> {
  %0 = mhlo.constant dense<[5, 66, 5, 1]> : tensor<4xi32>
  %1 = mhlo.constant dense<[3, 5, 1, 2]> : tensor<4xi32>
  // CHECK: mhlo.constant dense<[2, 1, 0, 1]>
  %2 = "mhlo.remainder"(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> (tensor<4xi32>)
  return %2 : tensor<4xi32>
}

// CHECK-LABEL: remainder_fold_float
func @remainder_fold_float() -> tensor<4xf32> {
  %0 = mhlo.constant dense<[7.0, 66.5, 5.0, 3.1]> : tensor<4xf32>
  %1 = mhlo.constant dense<[3.0, 5.0, 1.0, 2.6]> : tensor<4xf32>
  // CHECK: mhlo.constant dense<[1.000000e+00, 1.500000e+00, 0.000000e+00, 5.000000e-01]>
  %2 = "mhlo.remainder"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
  return %2 : tensor<4xf32>
}

// CHECK-LABEL: round_fold
func @round_fold() -> tensor<4xf32> {
  %0 = mhlo.constant dense<[-1.5, -0.1, 1.1, 2.5]> : tensor<4xf32>
  %1 = "mhlo.round_nearest_afz"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
  // CHECK: mhlo.constant dense<[-2.000000e+00, -0.000000e+00, 1.000000e+00, 3.000000e+00]>
}

// CHECK-LABEL: max_scalar_fold
func @max_scalar_fold() -> tensor<4xi64> {
  %0 = mhlo.constant dense<7> : tensor<4xi64>
  %1 = mhlo.constant dense<5> : tensor<4xi64>
  // CHECK: mhlo.constant dense<7>
  %2 = "mhlo.maximum"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  return %2 : tensor<4xi64>
}

// CHECK-LABEL: max_fold_float
func @max_fold_float() -> tensor<4xf64> {
  %0 = mhlo.constant dense<[5.0, 66.0, 5.0, 1.0]> : tensor<4xf64>
  %1 = mhlo.constant dense<[5.0, 3.0, 2.0, 4.0]> : tensor<4xf64>
  // CHECK: mhlo.constant dense<[5.000000e+00, 6.600000e+01, 5.000000e+00, 4.000000e+00]>
  %2 = "mhlo.maximum"(%0, %1) : (tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>)
  return %2 : tensor<4xf64>
}

// CHECK-LABEL: min_scalar_fold
func @min_scalar_fold() -> tensor<4xi64> {
  %0 = mhlo.constant dense<7> : tensor<4xi64>
  %1 = mhlo.constant dense<-5> : tensor<4xi64>
  // CHECK: mhlo.constant dense<-5>
  %2 = "mhlo.minimum"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  return %2 : tensor<4xi64>
}

// CHECK-LABEL: min_fold_float
func @min_fold_float() -> tensor<4xf64> {
  %0 = mhlo.constant dense<[5.0, 66.0, 5.0, 1.0]> : tensor<4xf64>
  %1 = mhlo.constant dense<[5.0, 3.0, 2.0, 4.0]> : tensor<4xf64>
  // CHECK: mhlo.constant dense<[5.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00]>
  %2 = "mhlo.minimum"(%0, %1) : (tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>)
  return %2 : tensor<4xf64>
}

// CHECK-LABEL: concatenate_noop
func @concatenate_noop(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK-SAME: [[ARG:%.+]]: tensor<4xi32>
  %0 = "mhlo.concatenate"(%arg0) { dimension = 0 : i64 } : (tensor<4xi32>) -> tensor<4xi32>

  // CHECK: return [[ARG]]
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: concatenate_remove_operand
func @concatenate_remove_operand(%arg0: tensor<4xi32>, %arg1: tensor<0xi32>) -> tensor<4xi32> {
  // CHECK-SAME: [[ARG0:%.+]]: tensor<4xi32>
  // CHECK-SAME: [[ARG1:%.+]]: tensor<0xi32>
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<4xi32>, tensor<0xi32>) -> tensor<4xi32>

  // CHECK: return [[ARG0]]
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: concatenate_empty_bool
func @concatenate_empty_bool(%arg0: tensor<0xi1>, %arg1: tensor<0xi1>) -> tensor<0xi1> {
  // CHECK: mhlo.constant
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<0xi1>, tensor<0xi1>) -> tensor<0xi1>

  return %0 : tensor<0xi1>
}

// CHECK-LABEL: concatenate_empty_int
func @concatenate_empty_int(%arg0: tensor<0xi32>, %arg1: tensor<0xi32>) -> tensor<0xi32> {
  // CHECK: mhlo.constant
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<0xi32>, tensor<0xi32>) -> tensor<0xi32>

  return %0 : tensor<0xi32>
}

// CHECK-LABEL: concatenate_empty_float
func @concatenate_empty_float(%arg0: tensor<0xf32>, %arg1: tensor<0xf32>) -> tensor<0xf32> {
  // CHECK: mhlo.constant
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<0xf32>, tensor<0xf32>) -> tensor<0xf32>

  return %0 : tensor<0xf32>
}

// CHECK-LABEL: concatenate_const_1D
func @concatenate_const_1D() -> tensor<4xi32> {
  // CHECK: [[VAL:%.+]]= mhlo.constant dense<[0, 1, 2, 3]>
  %0 = mhlo.constant dense<[0, 1]> : tensor<2xi32>
  %1 = mhlo.constant dense<[2, 3]> : tensor<2xi32>
  %2 = "mhlo.concatenate"(%0, %1) { dimension = 0 : i64 } : (tensor<2xi32>, tensor<2xi32>) -> tensor<4xi32>

  // CHECK: return [[VAL]]
  return %2 : tensor<4xi32>
}

// CHECK-LABEL: concatenate_const_1D_float
func @concatenate_const_1D_float() -> tensor<4xf32> {
  // CHECK: [[VAL:%.+]] = mhlo.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]>

  %0 = mhlo.constant dense<[0.0, 1.0]> : tensor<2xf32>
  %1 = mhlo.constant dense<[2.0, 3.0]> : tensor<2xf32>
  %2 = "mhlo.concatenate"(%0, %1) { dimension = 0 : i64 } : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>

  // CHECK: return [[VAL]]
  return %2 : tensor<4xf32>
}

// CHECK-LABEL: concatenate_const_2D_vertical
func @concatenate_const_2D_vertical() -> tensor<2x2xi32> {
  // CHECK: [[VAL:%.+]]= mhlo.constant dense<[
  // CHECK-SAME: [0, 1], [2, 3]
  // CHECK-SAME: ]>
  %0 = mhlo.constant dense<[[0, 1]]> : tensor<1x2xi32>
  %1 = mhlo.constant dense<[[2, 3]]> : tensor<1x2xi32>
  %2 = "mhlo.concatenate"(%0, %1) { dimension = 0 : i64 } : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>

  // CHECK: return [[VAL]]
  return %2 : tensor<2x2xi32>
}

// CHECK-LABEL: concatenate_const_2D_horizontal
func @concatenate_const_2D_horizontal() -> tensor<2x2xi32> {
  // CHECK: [[VAL:%.+]]= mhlo.constant dense<[
  // CHECK-SAME: [0, 2], [1, 3]
  // CHECK-SAME: ]>
  %0 = mhlo.constant dense<[[0], [1]]> : tensor<2x1xi32>
  %1 = mhlo.constant dense<[[2], [3]]> : tensor<2x1xi32>
  %2 = "mhlo.concatenate"(%0, %1) { dimension = 1 : i64 } : (tensor<2x1xi32>, tensor<2x1xi32>) -> tensor<2x2xi32>

  // CHECK: return [[VAL]]
  return %2 : tensor<2x2xi32>
}

// CHECK-LABEL: constant_like_constant
func @constant_like_constant(%arg0: tensor<3x4xi32>) -> tensor<3x4xf32> {
  // CHECK: mhlo.constant dense<3.200000e+00>
  %0 = "chlo.constant_like"(%arg0) { value = 3.2 : f32 } : (tensor<3x4xi32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// CHECK-LABEL: constant_like_constant_dynamic
func @constant_like_constant_dynamic(%arg0: tensor<*xi32>) -> tensor<*xf32> {
  // CHECK: chlo.constant_like
  %0 = "chlo.constant_like"(%arg0) { value = 3.2 : f32 } : (tensor<*xi32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: dynamic_slice_variable_start
func @dynamic_slice_variable_start(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  // CHECK: "mhlo.dynamic-slice"
  %1 = "mhlo.dynamic-slice"(%arg0, %arg1, %arg2) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  return %1 : tensor<1x4xi32>
}

// CHECK-LABEL: dynamic_slice_constant_start
func @dynamic_slice_constant_start(%arg0: tensor<4xi32>) -> tensor<2xi32> {
  // CHECK: %[[RESULT:.*]] =  "mhlo.slice"(%arg0)
  // CHECK-DAG-SAME: limit_indices = dense<3> : tensor<1xi64>
  // CHECK-DAG-SAME: start_indices = dense<1> : tensor<1xi64>
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>}
  // CHECK: return %[[RESULT]] : tensor<2xi32>
  %0 = mhlo.constant dense<1> : tensor<i64>
  %1 = "mhlo.dynamic-slice"(%arg0, %0) {slice_sizes = dense<2> : tensor<1xi64>} : (tensor<4xi32>, tensor<i64>) -> tensor<2xi32>
  return %1 : tensor<2xi32>
}

// CHECK-LABEL: dynamic_slice_constant_start_dynamic_shape
func @dynamic_slice_constant_start_dynamic_shape(%arg0: tensor<?x4xi32>, %arg1: tensor<2xi64>) -> tensor<?x4xi32> {
  // CHECK: %[[RESULT:.*]] = "mhlo.slice"(%arg0)
  // CHECK-DAG-SAME: limit_indices = dense<[2, 4]> : tensor<2xi64>
  // CHECK-DAG-SAME: start_indices = dense<[1, 0]> : tensor<2xi64>
  // CHECK-DAG-SAME: strides = dense<1> : tensor<2xi64>
  // CHECK: return %[[RESULT]] : tensor<?x4xi32>
  %0 = mhlo.constant dense<1> : tensor<i64>
  %1 = mhlo.constant dense<0> : tensor<i64>
  %2 = "mhlo.dynamic-slice"(%arg0, %0, %1) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<?x4xi32>, tensor<i64>, tensor<i64>) -> tensor<?x4xi32>
  return %2 : tensor<?x4xi32>
}

// CHECK-LABEL: slice_2D_noop
// CHECK-SAME: [[ARG:%.+]]: tensor<2x2xi64>
func @slice_2D_noop(%arg0: tensor<2x2xi64>) -> tensor<2x2xi64> {
  %0 = "mhlo.slice"(%arg0) { limit_indices = dense<[2, 2]> : tensor<2xi64>, start_indices = dense<[0, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xi64>) -> (tensor<2x2xi64>)

  // CHECK-NEXT: return [[ARG]]
  return %0 : tensor<2x2xi64>
}

// CHECK-LABEL: slice_1D_fold
func @slice_1D_fold() -> tensor<2xi64> {
  %0 = mhlo.constant dense<[5, 7, 9, 10]> : tensor<4xi64>
  // CHECK: mhlo.constant dense<[7, 9]>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[3]> : tensor<1xi64>, start_indices = dense<[1]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xi64>) -> (tensor<2xi64>)
  return %1 : tensor<2xi64>
}

// CHECK-LABEL: slice_1D_fp
func @slice_1D_fp() -> tensor<2xf32> {
  %0 = mhlo.constant dense<[5.0, 7.0, 9.0, 10.0]> : tensor<4xf32>
  // CHECK: mhlo.constant dense<[7.000000e+00, 9.000000e+00]>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[3]> : tensor<1xi64>, start_indices = dense<[1]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> (tensor<2xf32>)
  return %1 : tensor<2xf32>
}

// CHECK-LABEL: slice_1D_strided_fold
func @slice_1D_strided_fold() -> tensor<2xi64> {
  %0 = mhlo.constant dense<[5, 7, 9, 10]> : tensor<4xi64>
  // CHECK: mhlo.constant dense<[7, 10]>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[4]> : tensor<1xi64>, start_indices = dense<[1]> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>} : (tensor<4xi64>) -> (tensor<2xi64>)
  return %1 : tensor<2xi64>
}

// CHECK-LABEL: slice_2D_fold
func @slice_2D_fold() -> tensor<2x2xi64> {
  %0 = mhlo.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi64>
  // CHECK-NEXT: mhlo.constant dense<[
  // CHECK-SAME: [6, 7],
  // CHECK-SAME: [10, 11]
  // CHECK-SAME: ]>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[3, 4]> : tensor<2xi64>, start_indices = dense<[1, 2]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x4xi64>) -> (tensor<2x2xi64>)
  return %1 : tensor<2x2xi64>
}

// CHECK-LABEL: slice_2D_fold_horizontal
func @slice_2D_fold_horizontal() -> tensor<1x4xi64> {
  %0 = mhlo.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi64>
  // CHECK-NEXT: mhlo.constant dense<[
  // CHECK-SAME: [0, 1, 2, 3]
  // CHECK-SAME: ]>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[1, 4]> : tensor<2xi64>, start_indices = dense<[0, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x4xi64>) -> (tensor<1x4xi64>)
  return %1 : tensor<1x4xi64>
}

// CHECK-LABEL: slice_2D_fold_vertical
func @slice_2D_fold_vertical() -> tensor<4x1xi64> {
  %0 = mhlo.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi64>
  // CHECK-NEXT: mhlo.constant dense<[
  // CHECK-SAME: [2], [6], [10], [14]
  // CHECK-SAME: ]>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[4, 3]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x4xi64>) -> (tensor<4x1xi64>)
  return %1 : tensor<4x1xi64>
}

// CHECK-LABEL: slice_zero_elements
func @slice_zero_elements() -> tensor<0xi64> {
  %0 = mhlo.constant dense<> : tensor<0xi64>
  // CHECK: %[[CONST:.*]] = mhlo.constant dense<> : tensor<0xi64>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[0]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<0xi64>) -> (tensor<0xi64>)
  // CHECK: return %[[CONST]] : tensor<0xi64>
  return %1 : tensor<0xi64>
}

// CHECK-LABEL: slice_unknown_shape
func @slice_unknown_shape(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "mhlo.slice"(%arg0) {limit_indices = dense<[1, 4]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[1, 4]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: slice_concat_fold_first
func @slice_concat_fold_first(%arg0: tensor<1x5xf32>, %arg1: tensor<1x5xf32>) -> tensor<1x5xf32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<2x5xf32>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[1, 5]> : tensor<2xi64>, start_indices = dense<[0, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x5xf32>) -> (tensor<1x5xf32>)
  // CHECK: return %arg0
  return %1 : tensor<1x5xf32>
}

// CHECK-LABEL: slice_concat_fold_second
func @slice_concat_fold_second(%arg0: tensor<1x5xf32>, %arg1: tensor<1x5xf32>) -> tensor<1x5xf32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<2x5xf32>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[2, 5]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x5xf32>) -> (tensor<1x5xf32>)
  // CHECK: return %arg1
  return %1 : tensor<1x5xf32>
}

// CHECK-LABEL: slice_concat_fold_second_with_slice
func @slice_concat_fold_second_with_slice(%arg0: tensor<1x5xf32>, %arg1: tensor<1x5xf32>) -> tensor<1x4xf32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<2x5xf32>
  // CHECK: [[SLICE:%.+]] = "mhlo.slice"(%arg1) {limit_indices = dense<[1, 5]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x5xf32>) -> tensor<1x4xf32>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[2, 5]> : tensor<2xi64>, start_indices = dense<[1, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x5xf32>) -> (tensor<1x4xf32>)

  // CHECK: return [[SLICE]]
  return %1 : tensor<1x4xf32>
}

// CHECK-LABEL: slice_concat_fold_middle
func @slice_concat_fold_middle(%arg0: tensor<1x5xf32>, %arg1: tensor<2x5xf32>, %arg2: tensor<1x5xf32>) -> tensor<1x5xf32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1, %arg2) { dimension = 0 : i64 } : (tensor<1x5xf32>, tensor<2x5xf32>, tensor<1x5xf32>) -> tensor<4x5xf32>
  // CHECK: [[SLICE:%.+]] = "mhlo.slice"(%arg1) {limit_indices = dense<[2, 5]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[3, 5]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x5xf32>) -> (tensor<1x5xf32>)

  // CHECK: return [[SLICE]]
  return %1 : tensor<1x5xf32>
}

// CHECK-LABEL: slice_concat_fold_two
func @slice_concat_fold_two(%arg0: tensor<1x5xf32>, %arg1: tensor<2x5xf32>, %arg2: tensor<1x5xf32>) -> tensor<2x5xf32> {
  // CHECK: [[CONCAT:%.+]] = "mhlo.concatenate"(%arg1, %arg2) {dimension = 0 : i64}
  %0 = "mhlo.concatenate"(%arg0, %arg1, %arg2) { dimension = 0 : i64 } : (tensor<1x5xf32>, tensor<2x5xf32>, tensor<1x5xf32>) -> tensor<4x5xf32>

  // CHECK: [[SLICE:%.+]] = "mhlo.slice"([[CONCAT]]) {limit_indices = dense<[3, 5]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[4, 5]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x5xf32>) -> (tensor<2x5xf32>)

  // CHECK: return [[SLICE]]
  return %1 : tensor<2x5xf32>
}

// CHECK-LABEL: func @broadcast_in_dim_identity
func @broadcast_in_dim_identity(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  // CHECK: return %arg0
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func @broadcast_in_dim_not_identity_because_it_actually_broadcasts
func @broadcast_in_dim_not_identity_because_it_actually_broadcasts(%arg0: tensor<1x2xf32>) -> tensor<2x2xf32> {
  // CHECK: mhlo.broadcast_in_dim
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func @broadcast_in_dim_not_identity_permutation
func @broadcast_in_dim_not_identity_permutation(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: mhlo.broadcast_in_dim
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 0]> : tensor<2xi64>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}


// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic
func @dynamic_broadcast_in_dim_op_not_actually_dynamic(%arg0: tensor<4xf32>, %arg1: tensor<2xi64>) -> tensor<5x4xf32> {
  // CHECK: %[[RESULT:.+]] = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<5x4xf32>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %arg1) { broadcast_dimensions = dense<1> : tensor<1xi64> } : (tensor<4xf32>, tensor<2xi64>) -> tensor<5x4xf32>
  // CHECK: return %[[RESULT]] : tensor<5x4xf32>
  return %0 : tensor<5x4xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_to_same_shape_1
func @dynamic_broadcast_in_dim_to_same_shape_1(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<?xf32>
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<1xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %0) { broadcast_dimensions = dense<0> : tensor<1xi64> } : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK: return %[[ARG]] : tensor<?xf32>
  return %2 : tensor<?xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_to_same_shape_2
func @dynamic_broadcast_in_dim_to_same_shape_2(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<?xf32>
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> !shape.shape
  %1 = shape.to_extent_tensor %0 : !shape.shape -> tensor<1xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %1) { broadcast_dimensions = dense<0> : tensor<1xi64> } : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK: return %[[ARG]] : tensor<?xf32>
  return %2 : tensor<?xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_to_same_shape_3
func @dynamic_broadcast_in_dim_to_same_shape_3(%arg0: tensor<*xf32>) -> tensor<?xf32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<*xf32>
  %0 = shape.shape_of %arg0 : tensor<*xf32> -> tensor<?xindex>
  %1 = tensor.cast %0 : tensor<?xindex> to tensor<1xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %1) { broadcast_dimensions = dense<0> : tensor<1xi64> } : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK: %[[RES:.*]] = tensor.cast %[[ARG]] : tensor<*xf32> to tensor<?xf32>
  // CHECK: return %[[RES]] : tensor<?xf32>
  return %2 : tensor<?xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_to_same_shape_4
func @dynamic_broadcast_in_dim_to_same_shape_4(%arg0: tensor<*xf32>) -> tensor<?xf32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<*xf32>
  %0 = shape.shape_of %arg0 : tensor<*xf32> -> !shape.shape
  %1 = shape.to_extent_tensor %0 : !shape.shape -> tensor<?xindex>
  %2 = tensor.cast %1 : tensor<?xindex> to tensor<1xindex>
  %3 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %2) { broadcast_dimensions = dense<0> : tensor<1xi64> } : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK: %[[RES:.*]] = tensor.cast %[[ARG]] : tensor<*xf32> to tensor<?xf32>
  // CHECK: return %[[RES]] : tensor<?xf32>
  return %3 : tensor<?xf32>
}

// CHECK-LABEL: func @broadcast_in_dim_constant_fold_0d
func @broadcast_in_dim_constant_fold_0d() -> tensor<1x64x224x224xf32> {
  %cst = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %b = "mhlo.broadcast_in_dim"(%cst) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x64x224x224xf32>
  return %b : tensor<1x64x224x224xf32>
}
// CHECK-NEXT: %[[CST:.*]] = mhlo.constant dense<0.000000e+00> : tensor<1x64x224x224xf32>
// CHECK-NEXT: return %[[CST]] : tensor<1x64x224x224xf32>

// CHECK-LABEL: func @broadcast_in_dim_constant_fold
func @broadcast_in_dim_constant_fold() -> tensor<1x64x4x4xf32> {
  %cst = mhlo.constant dense<0.000000e+00> : tensor<4x4xf32>
  %b = "mhlo.broadcast_in_dim"(%cst) {broadcast_dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<4x4xf32>) -> tensor<1x64x4x4xf32>
  return %b : tensor<1x64x4x4xf32>
}
// CHECK-NEXT: %[[CST:.*]] = mhlo.constant dense<0.000000e+00> : tensor<1x64x4x4xf32>
// CHECK-NEXT: return %[[CST]] : tensor<1x64x4x4xf32>

// CHECK-LABEL: @complex_expand_fold
func @complex_expand_fold(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = "mhlo.complex"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xcomplex<f32>>)
  %1 = "mhlo.real"(%0) : (tensor<4xcomplex<f32>>) -> (tensor<4xf32>)
  %2 = "mhlo.imag"(%0) : (tensor<4xcomplex<f32>>) -> (tensor<4xf32>)
  // CHECK: return %arg0, %arg1
  return %1, %2 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: @complex_collapse_fold
func @complex_collapse_fold(%arg0: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %0 = "mhlo.real"(%arg0) : (tensor<4xcomplex<f32>>) -> (tensor<4xf32>)
  %1 = "mhlo.imag"(%arg0) : (tensor<4xcomplex<f32>>) -> (tensor<4xf32>)
  %2 = "mhlo.complex"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xcomplex<f32>>
  // CHECK: return %arg0
  return %2 : tensor<4xcomplex<f32>>
}

// CHECK-LABEL: @dynamic_iota_is_static
func @dynamic_iota_is_static(%arg0 : tensor<1xindex>) -> tensor<4xi32> {
  // CHECK: [[RESULT:%.*]] = "mhlo.iota"
  // CHECK: return [[RESULT]]
  %0 = "mhlo.dynamic_iota"(%arg0) {iota_dimension = 0 : i64} : (tensor<1xindex>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: @dynamic_iota_broadcast
func @dynamic_iota_broadcast(%arg0 : tensor<2xindex>) -> tensor<5x?xi32> {
  // CHECK: [[IOTA:%.+]] = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<5xi32>
  // CHECK: [[BROADCAST:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[IOTA]], %arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<5xi32>, tensor<2xindex>) -> tensor<5x?xi32>
  %0 = "mhlo.dynamic_iota"(%arg0) {iota_dimension = 0 : i64} : (tensor<2xindex>) -> tensor<5x?xi32>

  // CHECK: return [[BROADCAST]]
  return %0 : tensor<5x?xi32>
}

// CHECK-LABEL: @dynamic_iota_broadcast_second
func @dynamic_iota_broadcast_second(%arg0 : tensor<2xindex>) -> tensor<5x?xi32> {
  // CHECK-NEXT: [[CAST1:%.+]] = index_cast %arg0 : tensor<2xindex> to tensor<2xi64>
  // CHECK-NEXT: [[SLICE:%.+]] = "mhlo.slice"([[CAST1]]) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK-NEXT: [[CAST2:%.+]] = index_cast [[SLICE]] : tensor<1xi64> to tensor<1xindex>
  // CHECK-NEXT: [[IOTA:%.+]] = "mhlo.dynamic_iota"([[CAST2]]) {iota_dimension = 0 : i64} : (tensor<1xindex>) -> tensor<?xi32>
  // CHECK-NEXT: [[BROADCAST:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[IOTA]], %arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xi32>, tensor<2xindex>) -> tensor<5x?xi32>
  %0 = "mhlo.dynamic_iota"(%arg0) {iota_dimension = 1 : i64} : (tensor<2xindex>) -> tensor<5x?xi32>

  // CHECK: return [[BROADCAST]]
  return %0 : tensor<5x?xi32>
}

// CHECK-LABEL: @dynamic_iota_constant
func @dynamic_iota_constant(%arg0 : tensor<2xindex>) -> tensor<1x?xi32> {
  // CHECK: [[IOTA:%.+]] = mhlo.constant dense<0> : tensor<1xi32>
  // CHECK: [[BROADCAST:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[IOTA]], %arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi32>, tensor<2xindex>) -> tensor<1x?xi32>
  %0 = "mhlo.dynamic_iota"(%arg0) {iota_dimension = 0 : i64} : (tensor<2xindex>) -> tensor<1x?xi32>

  // CHECK: return [[BROADCAST]]
  return %0 : tensor<1x?xi32>
}

// CHECK-LABEL: @iota_constant
func @iota_constant() -> tensor<1xi32> {
  // CHECK: [[CONST:%.+]] = mhlo.constant dense<0> : tensor<1xi32>
  %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<1xi32>

  // CHECK: return [[CONST]] : tensor<1xi32>
  return %0 : tensor<1xi32>
}

// CHECK-LABEL: @iota_constant_multi
func @iota_constant_multi() -> tensor<1x4xi32> {
  // CHECK: [[CONST:%.+]] = mhlo.constant dense<0> : tensor<1x4xi32>
  %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<1x4xi32>

  // CHECK: return [[CONST]] : tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

// CHECK-LABEL: @iota_not_lowered_to_constant
func @iota_not_lowered_to_constant() -> tensor<4xi32> {
  // CHECK: [[RESULT:%.*]] = "mhlo.iota"
  // CHECK: return [[RESULT]]
  %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: @iota_broadcast
func @iota_broadcast() -> tensor<5x4xi32> {
  // CHECK: [[IOTA:%.+]] = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<5xi32>
  // CHECK: [[RESULT:%.+]] = "mhlo.broadcast_in_dim"([[IOTA]]) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<5xi32>) -> tensor<5x4xi32>
  %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<5x4xi32>

  return %0 : tensor<5x4xi32>
}

// CHECK-LABEL: @iota_broadcast
func @iota_broadcast_second() -> tensor<5x4xi32> {
  // CHECK: [[IOTA:%.+]] = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4xi32>
  // CHECK: [[RESULT:%.+]] = "mhlo.broadcast_in_dim"([[IOTA]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xi32>) -> tensor<5x4xi32>
  %0 = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<5x4xi32>

  return %0 : tensor<5x4xi32>
}

// CHECK-LABEL: @unary_einsum
func @unary_einsum(%arg0: tensor<2x3xf32>) -> tensor<2x2xf32> {
  // CHECK: %[[ONE:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: "mhlo.einsum"(%[[ONE]], %arg0) {einsum_config = ",ab->aa"}
  %0 = "mhlo.unary_einsum"(%arg0) {einsum_config = "ab->aa"} : (tensor<2x3xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func @fold_copy
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @fold_copy(%arg : tensor<1x4xf32>) -> tensor<1x4xf32> {
  // CHECK: return [[ARG]]
  %0 = "mhlo.copy"(%arg) : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// CHECK-LABEL: func @dynamic_reshape_not_actually_dynamic
func @dynamic_reshape_not_actually_dynamic(%arg0: tensor<4xf32>, %shape: tensor<2xindex>) -> tensor<4x1xf32> {
  // CHECK: mhlo.reshape
  %0 = "mhlo.dynamic_reshape"(%arg0, %shape) : (tensor<4xf32>, tensor<2xindex>) -> tensor<4x1xf32>
  return %0 : tensor<4x1xf32>
}

// CHECK-LABEL: func @shape_of_dynamic_reshape
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
func @shape_of_dynamic_reshape(%arg0: tensor<*xf32>, %shape: tensor<2xindex>) -> tensor<2xindex> {
  // CHECK: return [[ARG1]]
  %0 = "mhlo.dynamic_reshape"(%arg0, %shape) : (tensor<*xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %1 = shape.shape_of %0 : tensor<?x?xf32> -> tensor<2xindex>
  return %1 : tensor<2xindex>
}

// CHECK-LABEL: func @dynamic_reshape_rank_1_to_rank_1
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
func @dynamic_reshape_rank_1_to_rank_1(%arg0: tensor<?xcomplex<f32>>,
    %shape: tensor<?xindex>) -> tensor<?xf32> {
  // CHECK: [[RES:%[a-zA-Z0-9]+]] = "mhlo.real"([[ARG0]]) : (tensor<?xcomplex<f32>>) -> tensor<?xf32>
  // CHECK: return [[RES]]
  %0 = "mhlo.real"(%arg0): (tensor<?xcomplex<f32>>) -> tensor<?xf32>
  %1 = shape.shape_of %arg0 : tensor<?xcomplex<f32>> -> tensor<1xindex>
  %2 = shape.num_elements %1 : tensor<1xindex> -> index
  %3 = tensor.from_elements %2 : tensor<1xindex>
  %4 = "mhlo.dynamic_reshape"(%0, %3)
    : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
  return %4 : tensor<?xf32>
}

// CHECK-LABEL: func @dynamic_reshape_of_dynamic_reshape
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
func @dynamic_reshape_of_dynamic_reshape(%arg0: tensor<?xf16>, %shape: tensor<?xindex>) -> tensor<?xf16> {
  // CHECK: return [[ARG0]]
  %0 = "mhlo.dynamic_reshape"(%arg0, %shape) : (tensor<?xf16>, tensor<?xindex>) -> tensor<*xf16>
  %1 = shape.shape_of %0 : tensor<*xf16> -> tensor<?xindex>
  %2 = shape.num_elements %1 : tensor<?xindex> -> index
  %3 = tensor.from_elements %2 : tensor<1xindex>
  %4 = "mhlo.dynamic_reshape"(%0, %3) : (tensor<*xf16>, tensor<1xindex>) -> tensor<?xf16>
  return %4 : tensor<?xf16>
}

// CHECK-LABEL: do_not_dce_while_with_outfeed
func @do_not_dce_while_with_outfeed(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK: mhlo.while
  %0 = "mhlo.while"(%arg0) ( {
  ^bb0(%arg1: tensor<i64>):
    %1 = "mhlo.compare"(%arg1, %arg1) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i64>):
    %1 = "mhlo.create_token"() : () -> !mhlo.token
    // Side-effecting op outfeed present inside while.
    %2 = "mhlo.outfeed"(%arg1, %1) {outfeed_config = ""} : (tensor<i64>, !mhlo.token) -> !mhlo.token
    "mhlo.return"(%arg1) : (tensor<i64>) -> ()
  }) : (tensor<i64>) -> tensor<i64>

  return %arg0 : tensor<i64>
}

// CHECK-LABEL: dce_while_without_side_effect
func @dce_while_without_side_effect(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK-NOT: mhlo.while
  %0 = "mhlo.while"(%arg0) ( {
  ^bb0(%arg1: tensor<i64>):
    %1 = "mhlo.compare"(%arg1, %arg1) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i64>):
    %1 = "mhlo.create_token"() : () -> !mhlo.token
    "mhlo.return"(%arg1) : (tensor<i64>) -> ()
  }) : (tensor<i64>) -> tensor<i64>

  return %arg0 : tensor<i64>
}

// CHECK-LABEL: fold_compare_same_eq
func @fold_compare_same_eq(%arg0: tensor<i64>) -> tensor<i1> {
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %0 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = "EQ"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  return %0 : tensor<i1>
}

// CHECK-LABEL: fold_compare_same_le
func @fold_compare_same_le(%arg0: tensor<i64>) -> tensor<i1> {
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %0 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = "LE"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  return %0 : tensor<i1>
}

// CHECK-LABEL: fold_compare_same_ge
func @fold_compare_same_ge(%arg0: tensor<i64>) -> tensor<i1> {
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %0 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = "GE"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  return %0 : tensor<i1>
}
// CHECK-LABEL: fold_compare_same_ne
func @fold_compare_same_ne(%arg0: tensor<i64>) -> tensor<i1> {
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %0 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = "NE"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  return %0 : tensor<i1>
}

// CHECK-LABEL: fold_compare_same_lt
func @fold_compare_same_lt(%arg0: tensor<i64>) -> tensor<i1> {
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %0 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  return %0 : tensor<i1>
}

// CHECK-LABEL: fold_compare_same_gt
func @fold_compare_same_gt(%arg0: tensor<i64>) -> tensor<i1> {
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %0 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = "GT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  return %0 : tensor<i1>
}

// Address NaN != NaN.
// CHECK-LABEL: dont_fold_compare_same_eq_float
func @dont_fold_compare_same_eq_float(%arg0: tensor<f16>) -> tensor<i1> {
  // CHECK: %0 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = "EQ"} : (tensor<f16>, tensor<f16>) -> tensor<i1>
  %0 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = "EQ"} : (tensor<f16>, tensor<f16>) -> tensor<i1>
  return %0 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_eq
func @fold_compare_false_eq() -> tensor<i1> {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "EQ"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %2 : tensor<i1>
}
// CHECK-LABEL: fold_compare_true_eq
func @fold_compare_true_eq() -> tensor<i1> {
  %0 = mhlo.constant dense<1> : tensor<i32>
  %1 = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "EQ"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_eq_float
func @fold_compare_false_eq_float() -> tensor<i1> {
  %0 = mhlo.constant dense<0.> : tensor<f32>
  %1 = mhlo.constant dense<1.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "EQ"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_eq_float
func @fold_compare_true_eq_float() -> tensor<i1> {
  %0 = mhlo.constant dense<1.> : tensor<f32>
  %1 = mhlo.constant dense<1.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "EQ"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_ne
func @fold_compare_false_ne() -> tensor<i1> {
  %0 = mhlo.constant dense<1> : tensor<i32>
  %1 = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "NE"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_ne
func @fold_compare_true_ne() -> tensor<i1> {
  %0 = mhlo.constant dense<1> : tensor<i32>
  %1 = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "NE"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_ne_float
func @fold_compare_false_ne_float() -> tensor<i1> {
  %0 = mhlo.constant dense<1.> : tensor<f32>
  %1 = mhlo.constant dense<1.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "NE"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_ne_float
func @fold_compare_true_ne_float() -> tensor<i1> {
  %0 = mhlo.constant dense<0.> : tensor<f32>
  %1 = mhlo.constant dense<1.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "NE"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_lt
func @fold_compare_false_lt() -> tensor<i1> {
  %0 = mhlo.constant dense<1> : tensor<i32>
  %1 = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_lt
func @fold_compare_true_lt() -> tensor<i1> {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_lt_float
func @fold_compare_false_lt_float() -> tensor<i1> {
  %0 = mhlo.constant dense<1.> : tensor<f32>
  %1 = mhlo.constant dense<1.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "LT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_lt_float
func @fold_compare_true_lt_float() -> tensor<i1> {
  %0 = mhlo.constant dense<0.> : tensor<f32>
  %1 = mhlo.constant dense<1.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "LT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_le
func @fold_compare_false_le() -> tensor<i1> {
  %0 = mhlo.constant dense<1> : tensor<i32>
  %1 = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "LE"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_le
func @fold_compare_true_le() -> tensor<i1> {
  %0 = mhlo.constant dense<1> : tensor<i32>
  %1 = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "LE"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_le_float
func @fold_compare_false_le_float() -> tensor<i1> {
  %0 = mhlo.constant dense<1.> : tensor<f32>
  %1 = mhlo.constant dense<0.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "LE"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_le_float
func @fold_compare_true_le_float() -> tensor<i1> {
  %0 = mhlo.constant dense<1.> : tensor<f32>
  %1 = mhlo.constant dense<1.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "LE"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_gt
func @fold_compare_false_gt() -> tensor<i1> {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "GT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_gt
func @fold_compare_true_gt() -> tensor<i1> {
  %0 = mhlo.constant dense<1> : tensor<i32>
  %1 = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "GT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_gt_float
func @fold_compare_false_gt_float() -> tensor<i1> {
  %0 = mhlo.constant dense<0.> : tensor<f32>
  %1 = mhlo.constant dense<0.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_gt_float
func @fold_compare_true_gt_float() -> tensor<i1> {
  %0 = mhlo.constant dense<1.> : tensor<f32>
  %1 = mhlo.constant dense<0.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_ge
func @fold_compare_false_ge() -> tensor<i1> {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "GE"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_ge
func @fold_compare_true_ge() -> tensor<i1> {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "GE"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_ge_float
func @fold_compare_false_ge_float() -> tensor<i1> {
  %0 = mhlo.constant dense<0.> : tensor<f32>
  %1 = mhlo.constant dense<1.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "GE"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_ge_float
func @fold_compare_true_ge_float() -> tensor<i1> {
  %0 = mhlo.constant dense<0.> : tensor<f32>
  %1 = mhlo.constant dense<0.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = "GE"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: unpack_repack_same_tuple
// CHECK-SAME: ([[ARG0:%.*]]: tuple<tensor<i32>, !mhlo.token, tensor<f32>>)
func @unpack_repack_same_tuple(%arg0: tuple<tensor<i32>, !mhlo.token, tensor<f32>>) -> tuple<tensor<i32>, !mhlo.token, tensor<f32>> {
  %0 = "mhlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<i32>, !mhlo.token, tensor<f32>>) -> tensor<i32>
  %1 = "mhlo.get_tuple_element"(%arg0) {index = 1 : i32} : (tuple<tensor<i32>, !mhlo.token, tensor<f32>>) -> !mhlo.token
  %2 = "mhlo.get_tuple_element"(%arg0) {index = 2 : i32} : (tuple<tensor<i32>, !mhlo.token, tensor<f32>>) -> tensor<f32>
  %3 = "mhlo.tuple"(%0, %1, %2) : (tensor<i32>, !mhlo.token, tensor<f32>) -> tuple<tensor<i32>, !mhlo.token, tensor<f32>>

  // CHECK: return [[ARG0]]
  return %3 : tuple<tensor<i32>, !mhlo.token, tensor<f32>>
}

// CHECK-LABEL: unpack_repack_same_tuple_single_element
// CHECK-SAME: ([[ARG0:%.*]]: tuple<tensor<i32>>)
func @unpack_repack_same_tuple_single_element(%arg0: tuple<tensor<i32>>) -> tuple<tensor<i32>> {
  %0 = "mhlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<i32>>) -> tensor<i32>
  %3 = "mhlo.tuple"(%0) : (tensor<i32>) -> tuple<tensor<i32>>

  // CHECK: return [[ARG0]]
  return %3 : tuple<tensor<i32>>
}

// CHECK-LABEL: func @erase_dead_lhlo_constant
func @erase_dead_lhlo_constant() {
  %M = alloc() : memref<256x1024xf32>
  // CHECK-NEXT: return
  "lmhlo.constant"(%M) {value = dense<0.0> : tensor<f32>} : (memref<256x1024xf32>) -> ()
  dealloc %M : memref<256x1024xf32>
  return
}

// A negative test for dead lhlo constant op erasure.
// CHECK-LABEL: func @erase_dead_lhlo_constant_negative
func @erase_dead_lhlo_constant_negative(%M : memref<4xf32>) -> memref<256x1024xf32> {
  // CHECK-NEXT: lmhlo.constant
  "lmhlo.constant"(%M) {value = dense<0.0> : tensor<f32>} : (memref<4xf32>) -> ()
  // CHECK-NEXT: alloc
  // CHECK-NEXT: lmhlo.constant
  %N = alloc() : memref<256x1024xf32>
  "lmhlo.constant"(%N) {value = dense<0.0> : tensor<f32>} : (memref<256x1024xf32>) -> ()
  return %N : memref<256x1024xf32>
}

// CHECK-LABEL: func @fold_get_dimension_size
func @fold_get_dimension_size(%I: tensor<1x128x512xf32>) -> tensor<i32> {
  %size = "mhlo.get_dimension_size"(%I) {dimension = 2 : i64} : (tensor<1x128x512xf32>) -> tensor<i32>
  return %size : tensor<i32>
  // CHECK-NEXT: %[[C:.*]] = mhlo.constant dense<512> : tensor<i32>
  // CHECK-NEXT: return %[[C]]
}

// CHECK-LABEL: func @fold_set_dimension_size
// CHECK-SAME: (%[[I:.*]]: tensor<1x128x512xf32>)
func @fold_set_dimension_size(%I: tensor<1x128x512xf32>) -> tensor<1x128x512xf32> {
  %dim = mhlo.constant dense<512> : tensor<i32>
  %result = "mhlo.set_dimension_size"(%I, %dim) {dimension = 2 : i64} : (tensor<1x128x512xf32>, tensor<i32>) -> tensor<1x128x512xf32>
  return %result : tensor<1x128x512xf32>

  // CHECK-NEXT: return %[[I]]
}

// CHECK-LABEL: func @fold_select_same
func @fold_select_same(%arg0 : tensor<f32>, %arg1 : tensor<i1>) -> tensor<f32> {
  %1 = "mhlo.select"(%arg1, %arg0, %arg0) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: return %arg0
  return %1 : tensor<f32>
}

// CHECK-LABEL: func @fold_select_first
func @fold_select_first(%arg0 : tensor<f32>, %arg1 : tensor<f32>) -> tensor<f32> {
  %0 = mhlo.constant dense<1> : tensor<i1>
  %1 = "mhlo.select"(%0, %arg0, %arg1) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: return %arg0
  return %1 : tensor<f32>
}

// CHECK-LABEL: func @fold_select_second
func @fold_select_second(%arg0 : tensor<f32>, %arg1 : tensor<f32>) -> tensor<f32> {
  %0 = mhlo.constant dense<0> : tensor<i1>
  %1 = "mhlo.select"(%0, %arg0, %arg1) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: return %arg1
  return %1 : tensor<f32>
}

// CHECK-LABEL: func @fold_select_vector
func @fold_select_vector(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = mhlo.constant dense<1> : tensor<4xi1>
  %1 = "mhlo.select"(%0, %arg0, %arg1) : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK: return %arg0
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: gather_to_slice
func @gather_to_slice(%arg0: tensor<5x6x7xf32>) -> tensor<3x6x5xf32> {
  %0 = constant dense<[1, 2]> : tensor<2xi32>
  %1 = "mhlo.gather"(%arg0, %0) {
    dimension_numbers = {collapsed_slice_dims = dense<> : tensor<0xi64>,
                         index_vector_dim = 0 : i64,
                         offset_dims = dense<[0, 1, 2]> : tensor<3xi64>,
                         start_index_map = dense<[0, 2]> : tensor<2xi64>},
    indices_are_sorted = false,
    slice_sizes = dense<[3, 6, 5]> : tensor<3xi64>} : (tensor<5x6x7xf32>, tensor<2xi32>) -> tensor<3x6x5xf32>
  return %1 : tensor<3x6x5xf32>
  // CHECK:  %[[RET:.*]] = "mhlo.slice"(%arg0) {limit_indices = dense<[4, 6, 7]> : tensor<3xi64>, start_indices = dense<[1, 0, 2]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<5x6x7xf32>) -> tensor<3x6x5xf32>
  // CHECK: return %[[RET]] : tensor<3x6x5xf32>
}

// CHECK-LABEL: gather_scalar_index_to_slice
func @gather_scalar_index_to_slice(%arg0: tensor<5x6x7xf32>) -> tensor<5x6x4xf32> {
  %0 = constant dense<1> : tensor<i32>
  %1 = "mhlo.gather"(%arg0, %0) {
    dimension_numbers = {collapsed_slice_dims = dense<> : tensor<0xi64>,
                         index_vector_dim = 0 : i64,
                         offset_dims = dense<[0, 1, 2]> : tensor<3xi64>,
                         start_index_map = dense<[2]> : tensor<1xi64>},
    indices_are_sorted = false,
    slice_sizes = dense<[5, 6, 4]> : tensor<3xi64>} : (tensor<5x6x7xf32>, tensor<i32>) -> tensor<5x6x4xf32>
  return %1 : tensor<5x6x4xf32>
  // CHECK:  %[[RET:.*]] = "mhlo.slice"(%arg0) {limit_indices = dense<[5, 6, 5]> : tensor<3xi64>, start_indices = dense<[0, 0, 1]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<5x6x7xf32>) -> tensor<5x6x4xf32>
  // CHECK: return %[[RET]] : tensor<5x6x4xf32>
}

// CHECK-LABEL: gather_to_slice_reshape
func @gather_to_slice_reshape(%arg0: tensor<5x6x7xf32>) -> tensor<3x6xf32> {
  %0 = constant dense<[1, 2]> : tensor<2xi32>
  %1 = "mhlo.gather"(%arg0, %0) {
    dimension_numbers = {collapsed_slice_dims = dense<[2]> : tensor<1xi64>,
                         index_vector_dim = 0 : i64,
                         offset_dims = dense<[0, 1, 2]> : tensor<3xi64>,
                         start_index_map = dense<[0, 2]> : tensor<2xi64>},
    indices_are_sorted = false,
    slice_sizes = dense<[3, 6, 1]> : tensor<3xi64>} : (tensor<5x6x7xf32>, tensor<2xi32>) -> tensor<3x6xf32>
  return %1 : tensor<3x6xf32>
  // CHECK:  %[[V0:.*]] = "mhlo.slice"(%arg0) {limit_indices = dense<[4, 6, 3]> : tensor<3xi64>, start_indices = dense<[1, 0, 2]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<5x6x7xf32>) -> tensor<3x6x1xf32>
  // CHECK:  %[[V1:.*]] = "mhlo.reshape"(%[[V0]]) : (tensor<3x6x1xf32>) -> tensor<3x6xf32>
  // CHECK: return %[[V1]] : tensor<3x6xf32>
}

// CHECK-LABEL: func @fold_and_same
func @fold_and_same(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = "mhlo.and"(%arg0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: return %arg0
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_and_ones
func @fold_and_ones(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<-1> : tensor<4xi32>
  %1 = "mhlo.and"(%0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: return %arg0
  return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_and_zeros
func @fold_and_zeros(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<0> : tensor<4xi32>
  %1 = "mhlo.and"(%0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: return %0
  return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_and_constant
func @fold_and_constant(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<7> : tensor<4xi32>
  // CHECK: mhlo.and
  %1 = "mhlo.and"(%0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_and_constants
func @fold_and_constants() -> tensor<4xi32> {
  %0 = mhlo.constant dense<[0, 1, 6, 3]> : tensor<4xi32>
  %1 = mhlo.constant dense<[7, 3, 7, 2]> : tensor<4xi32>
  %2 = "mhlo.and"(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: %0 = mhlo.constant dense<[0, 1, 6, 2]> : tensor<4xi32>
  // CHECK: return %0
  return %2 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_or_same
func @fold_or_same(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = "mhlo.or"(%arg0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: return %arg0
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_or_ones
func @fold_or_ones(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<-1> : tensor<4xi32>
  %1 = "mhlo.or"(%0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: return %0
  return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_or_zeros
func @fold_or_zeros(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<0> : tensor<4xi32>
  %1 = "mhlo.or"(%0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: return %arg0
  return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_or_constant
func @fold_or_constant(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<7> : tensor<4xi32>
  // CHECK: mhlo.or
  %1 = "mhlo.or"(%0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_or_zeros_right
func @fold_or_zeros_right(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<0> : tensor<4xi32>
  %1 = "mhlo.or"(%arg0, %0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: return %arg0
  return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_or_zeros_constants
func @fold_or_zeros_constants() -> tensor<4xi32> {
  %0 = mhlo.constant dense<[0, 1, 6, 3]> : tensor<4xi32>
  %1 = mhlo.constant dense<[7, 3, 7, 2]> : tensor<4xi32>
  %2 = "mhlo.or"(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: %0 = mhlo.constant dense<[7, 3, 7, 3]> : tensor<4xi32>
  // CHECK: return %0
  return %2 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_xor_same
func @fold_xor_same(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = "mhlo.xor"(%arg0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: %0 = mhlo.constant dense<0> : tensor<4xi32>
  // CHECK: return %0
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_xor_ones_left
func @fold_xor_ones_left(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<-1> : tensor<4xi32>
  // CHECK: mhlo.xor
  %1 = "mhlo.xor"(%0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_xor_ones_right
func @fold_xor_ones_right(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<-1> : tensor<4xi32>
  // CHECK: mhlo.xor
  %1 = "mhlo.xor"(%arg0, %0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_xor_zeros_left
func @fold_xor_zeros_left(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<0> : tensor<4xi32>
  %1 = "mhlo.xor"(%0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: return %arg0
  return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_xor_zeros_right
func @fold_xor_zeros_right(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<0> : tensor<4xi32>
  %1 = "mhlo.xor"(%arg0, %0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: return %arg0
  return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_xor_zeros_constants
func @fold_xor_zeros_constants() -> tensor<4xi32> {
  %0 = mhlo.constant dense<[0, 1, 6, 3]> : tensor<4xi32>
  %1 = mhlo.constant dense<[7, 3, 7, 2]> : tensor<4xi32>
  %2 = "mhlo.xor"(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: %0 = mhlo.constant dense<[7, 2, 1, 1]> : tensor<4xi32>
  // CHECK: return %0
  return %2 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_negate_int
func @fold_negate_int() -> tensor<4xi32> {
  %0 = mhlo.constant dense<[0, 1, 6, -3]> : tensor<4xi32>
  // CHECK: mhlo.constant dense<[0, -1, -6, 3]>
  %1 = "mhlo.negate"(%0) : (tensor<4xi32>) -> tensor<4xi32>
  return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_negate_float
func @fold_negate_float() -> tensor<4xf32> {
  %0 = mhlo.constant dense<[0., 1., 6., -3.]> : tensor<4xf32>
  // CHECK: mhlo.constant dense<[-0.000000e+00, -1.000000e+00, -6.000000e+00, 3.000000e+00]>
  %1 = "mhlo.negate"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func @fold_sqrt_f32_constants
func @fold_sqrt_f32_constants() -> tensor<4xf32> {
  %0 = mhlo.constant dense<1.0> : tensor<4xf32>
  %1 = "mhlo.sqrt"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  //     CHECK: mhlo.constant dense<1.000000e+00> : tensor<4xf32>
  // CHECK-NOT: mhlo.sqrt
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: func @fold_sqrt_f64_constants
func @fold_sqrt_f64_constants() -> tensor<4xf64> {
  %0 = mhlo.constant dense<[1.0, 4.0, 9.0, 16.0]> : tensor<4xf64>
  %1 = "mhlo.sqrt"(%0) : (tensor<4xf64>) -> tensor<4xf64>
  //     CHECK: mhlo.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf64>
  // CHECK-NOT: mhlo.sqrt
  return %1 : tensor<4xf64>
}

// CHECK-LABEL: func @not_fold_sqrt_neg_constants
func @not_fold_sqrt_neg_constants() -> tensor<4xf32> {
  %0 = mhlo.constant dense<-1.0> : tensor<4xf32>
  %1 = "mhlo.sqrt"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: mhlo.constant dense<-1.000000e+00> : tensor<4xf32>
  // CHECK: mhlo.sqrt
  return %1 : tensor<4xf32>
}

// CHECK-LABEL: @tensor_flow_scatter_v1_update
func @tensor_flow_scatter_v1_update() -> tensor<3x3xi32> {
  %0 = constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi32>
  %1 = constant dense<[0, 2]> : tensor<2xi32>
  %2 = constant dense<[[10, 20, 30], [70, 80, 90]]> : tensor<2x3xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ( {
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      "mhlo.return"(%arg1) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers = {
          index_vector_dim = 1 : i64,
          inserted_window_dims = dense<0> : tensor<1xi64>,
          scatter_dims_to_operand_dims = dense<0> : tensor<1xi64>,
          update_window_dims = dense<[1]> : tensor<1xi64>
        },
        unique_indices = false
    } : (tensor<3x3xi32>, tensor<2xi32>, tensor<2x3xi32>) -> tensor<3x3xi32>
  return %3 : tensor<3x3xi32>
  // CHECK: mhlo.constant dense<[
  // CHECK-SAME: [10, 20, 30], [4, 5, 6], [70, 80, 90]
  // CHECK-SAME: ]> : tensor<3x3xi32>
}

// CHECK-LABEL: @tensor_flow_scatter_v2_update
func @tensor_flow_scatter_v2_update() -> tensor<3x3xi32> {
  %0 = constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi32>
  %1 = constant dense<[0, 2]> : tensor<2xi32>
  %2 = constant dense<[[10, 30], [40, 60], [70, 90]]> : tensor<3x2xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ( {
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      "mhlo.return"(%arg1) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers = {
          index_vector_dim = 1 : i64,
          inserted_window_dims = dense<1> : tensor<1xi64>,
          scatter_dims_to_operand_dims = dense<1> : tensor<1xi64>,
          update_window_dims = dense<[0]> : tensor<1xi64>
        },
        unique_indices = false
    } : (tensor<3x3xi32>, tensor<2xi32>, tensor<3x2xi32>) -> tensor<3x3xi32>
  return %3 : tensor<3x3xi32>
  // CHECK: mhlo.constant dense<[
  // CHECK-SAME: [10, 2, 30], [40, 5, 60], [70, 8, 90]
  // CHECK-SAME: ]> : tensor<3x3xi32>
}

// CHECK-LABEL: @tensor_flow_scatter_add
func @tensor_flow_scatter_add() -> tensor<3x3xi32> {
  %0 = constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi32>
  %1 = constant dense<[0, 2]> : tensor<2xi32>
  %2 = constant dense<[[10, 20, 30], [70, 80, 90]]> : tensor<2x3xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ( {
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %4 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
      "mhlo.return"(%4) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers = {
          index_vector_dim = 1 : i64,
          inserted_window_dims = dense<0> : tensor<1xi64>,
          scatter_dims_to_operand_dims = dense<0> : tensor<1xi64>,
          update_window_dims = dense<[1]> : tensor<1xi64>
        },
        unique_indices = false
    } : (tensor<3x3xi32>, tensor<2xi32>, tensor<2x3xi32>) -> tensor<3x3xi32>
  return %3 : tensor<3x3xi32>
  // CHECK: mhlo.constant dense<[
  // CHECK-SAME: [11, 22, 33], [4, 5, 6], [77, 88, 99]
  // CHECK-SAME: ]> : tensor<3x3xi32>
}

// CHECK-LABEL: @tensor_flow_scatter_repeated
func @tensor_flow_scatter_repeated() -> tensor<3x3xi32> {
  %0 = constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi32>
  %1 = constant dense<[1, 1]> : tensor<2xi32>
  %2 = constant dense<[[10, 20, 30], [70, 80, 90]]> : tensor<2x3xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ( {
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %4 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
      "mhlo.return"(%4) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers = {
          index_vector_dim = 1 : i64,
          inserted_window_dims = dense<0> : tensor<1xi64>,
          scatter_dims_to_operand_dims = dense<0> : tensor<1xi64>,
          update_window_dims = dense<[1]> : tensor<1xi64>
        },
        unique_indices = false
    } : (tensor<3x3xi32>, tensor<2xi32>, tensor<2x3xi32>) -> tensor<3x3xi32>
  return %3 : tensor<3x3xi32>
  // CHECK: mhlo.constant dense<[
  // CHECK-SAME: [1, 2, 3], [84, 105, 126], [7, 8, 9]
  // CHECK-SAME: ]> : tensor<3x3xi32>
}

// CHECK-LABEL: @tensor_flow_scatter_multiple_batch
func @tensor_flow_scatter_multiple_batch() -> tensor<3x3xi32> {
  %0 = constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi32>
  %1 = constant dense<[[0, 2], [2, 1]]> : tensor<2x2xi32>
  %2 = constant dense<[[[10, 30], [40, 60], [70, 90]], [[5, 5], [5, 5], [5, 5]]]> : tensor<2x3x2xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ( {
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %4 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
      "mhlo.return"(%4) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers = {
          index_vector_dim = 2 : i64,
          inserted_window_dims = dense<1> : tensor<1xi64>,
          scatter_dims_to_operand_dims = dense<1> : tensor<1xi64>,
          update_window_dims = dense<[1]> : tensor<1xi64>
        },
        unique_indices = false
    } : (tensor<3x3xi32>, tensor<2x2xi32>, tensor<2x3x2xi32>) -> tensor<3x3xi32>
  return %3 : tensor<3x3xi32>
  // CHECK: mhlo.constant dense<[
  // CHECK-SAME: [11, 7, 38], [44, 10, 71], [77, 13, 104]
  // CHECK-SAME: ]> : tensor<3x3xi32>
}

// CHECK-LABEL: @tensor_flow_scatter_nd
func @tensor_flow_scatter_nd() -> tensor<3x3x2xi32> {
  %0 = constant dense<[[[-1, 1], [-2, 2], [-3, 3]], [[-4, 4], [-5, 5], [-6, 6]], [[-7, 7], [-8, 8], [-9, 9]]]> : tensor<3x3x2xi32>
  %1 = constant dense<[[0, 0], [1, 0]]> : tensor<2x2xi32>
  %2 = constant dense<[[-10, 10], [-40, 40]]> : tensor<2x2xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ( {
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      "mhlo.return"(%arg1) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers = {
          index_vector_dim = 1 : i64,
          inserted_window_dims = dense<[0, 1]> : tensor<2xi64>,
          scatter_dims_to_operand_dims = dense<[0, 1]> : tensor<2xi64>,
          update_window_dims = dense<[1]> : tensor<1xi64>
        },
        unique_indices = false
    } : (tensor<3x3x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<3x3x2xi32>
  return %3 : tensor<3x3x2xi32>
  // CHECK: mhlo.constant dense<[
  // CHECK-SAME: [-10, 10], [-2, 2], [-3, 3]
  // CHECK-SAME: [-40, 40], [-5, 5], [-6, 6]
  // CHECK-SAME: [-7, 7], [-8, 8], [-9, 9]
  // CHECK-SAME: ]> : tensor<3x3x2xi32>
}

// CHECK-LABEL: @tensor_flow_scatter_nd_index_vector
func @tensor_flow_scatter_nd_index_vector() -> tensor<3x3x2xi32> {
  %0 = constant dense<[[[-1, 1], [-2, 2], [-3, 3]], [[-4, 4], [-5, 5], [-6, 6]], [[-7, 7], [-8, 8], [-9, 9]]]> : tensor<3x3x2xi32>
  %1 = constant dense<[[0, 0], [1, 0]]> : tensor<2x2xi32>
  %2 = constant dense<[[-10, 10], [-20, 20]]> : tensor<2x2xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ( {
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      "mhlo.return"(%arg1) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers = {
          index_vector_dim = 0 : i64,
          inserted_window_dims = dense<[0, 1]> : tensor<2xi64>,
          scatter_dims_to_operand_dims = dense<[0, 1]> : tensor<2xi64>,
          update_window_dims = dense<[1]> : tensor<1xi64>
        },
        unique_indices = false
    } : (tensor<3x3x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<3x3x2xi32>
  return %3 : tensor<3x3x2xi32>
  // CHECK: mhlo.constant dense<[
  // CHECK-SAME: [-20, 20], [-10, 10], [-3, 3]
  // CHECK-SAME: [-4, 4], [-5, 5], [-6, 6]
  // CHECK-SAME: [-7, 7], [-8, 8], [-9, 9]
  // CHECK-SAME: ]> : tensor<3x3x2xi32>
}

// CHECK-LABEL: @scatter_batch_dus
func @scatter_batch_dus() -> tensor<3x3xi32> {
  %0 = constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi32>
  %1 = constant dense<[[2, 1], [1, 1]]> : tensor<2x2xi32>
  %2 = constant dense<[[[10]], [[20]]]> : tensor<2x1x1xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ( {
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      "mhlo.return"(%arg1) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers = {
          index_vector_dim = 0 : i64,
          inserted_window_dims = dense<> : tensor<0xi64>,
          scatter_dims_to_operand_dims = dense<[0, 1]> : tensor<2xi64>,
          update_window_dims = dense<[1, 2]> : tensor<2xi64>
        },
        unique_indices = false
    } : (tensor<3x3xi32>, tensor<2x2xi32>, tensor<2x1x1xi32>) -> tensor<3x3xi32>
  return %3 : tensor<3x3xi32>
  // CHECK: mhlo.constant dense<[
  // CHECK-SAME: [1, 2, 3], [4, 20, 6], [7, 10, 9]
  // CHECK-SAME: ]> : tensor<3x3xi32>
}

// CHECK-LABEL: @scatter_no_update_window_dim
func @scatter_no_update_window_dim() -> tensor<3xi32> {
  %0 = constant dense<[0, 1, 2]> : tensor<3xi32>
  %1 = constant dense<[[[0], [1]], [[2], [1]]]> : tensor<2x2x1xi32>
  %2 = constant dense<[[10, 20], [30, 40]]> : tensor<2x2xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ( {
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %4 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
      "mhlo.return"(%4) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers = {
          index_vector_dim = 2 : i64,
          inserted_window_dims = dense<0> : tensor<1xi64>,
          scatter_dims_to_operand_dims = dense<0> : tensor<1xi64>,
          update_window_dims = dense<> : tensor<0xi64>
        },
        unique_indices = false
    } : (tensor<3xi32>, tensor<2x2x1xi32>, tensor<2x2xi32>) -> tensor<3xi32>
  return %3 : tensor<3xi32>
  // CHECK: mhlo.constant dense<[10, 61, 32]> : tensor<3xi32>
}

// CHECK-LABEL: @scatter_negative_index
func @scatter_negative_index() -> tensor<3x3xi32> {
  %0 = constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi32>
  %1 = constant dense<[0, -1]> : tensor<2xi32>
  %2 = constant dense<[[10, 20, 30], [70, 80, 90]]> : tensor<2x3xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ( {
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      "mhlo.return"(%arg1) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers = {
          index_vector_dim = 1 : i64,
          inserted_window_dims = dense<0> : tensor<1xi64>,
          scatter_dims_to_operand_dims = dense<0> : tensor<1xi64>,
          update_window_dims = dense<[1]> : tensor<1xi64>
        },
        unique_indices = false
    } : (tensor<3x3xi32>, tensor<2xi32>, tensor<2x3xi32>) -> tensor<3x3xi32>
  return %3 : tensor<3x3xi32>
  // CHECK: constant dense<[
  // CHECK-SAME: [1, 2, 3], [4, 5, 6], [7, 8, 9]
  // CHECK-SAME: ]> : tensor<3x3xi32>
  // CHECK: "mhlo.scatter"
}

// CHECK-LABEL: @scatter_out_of_bound
func @scatter_out_of_bound() -> tensor<3x3xi32> {
  %0 = constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi32>
  %1 = constant dense<[1, 5]> : tensor<2xi32>
  %2 = constant dense<[[10, 20, 30], [70, 80, 90]]> : tensor<2x3xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ( {
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      "mhlo.return"(%arg1) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers = {
          index_vector_dim = 1 : i64,
          inserted_window_dims = dense<0> : tensor<1xi64>,
          scatter_dims_to_operand_dims = dense<0> : tensor<1xi64>,
          update_window_dims = dense<[1]> : tensor<1xi64>
        },
        unique_indices = false
    } : (tensor<3x3xi32>, tensor<2xi32>, tensor<2x3xi32>) -> tensor<3x3xi32>
  return %3 : tensor<3x3xi32>
  // CHECK: constant dense<[
  // CHECK-SAME: [1, 2, 3], [4, 5, 6], [7, 8, 9]
  // CHECK-SAME: ]> : tensor<3x3xi32>
  // CHECK: "mhlo.scatter"
}

// CHECK-LABEL: @pad_identity_fold
func @pad_identity_fold(%arg0: tensor<5x7xf32>) -> tensor<5x7xf32> {
  %0 = constant dense<0.0> : tensor<f32>
  %1 = "mhlo.pad"(%arg0, %0) {
    edge_padding_low = dense<0> : tensor<2xi64>,
    edge_padding_high = dense<0> : tensor<2xi64>,
    interior_padding = dense<0> : tensor<2xi64>
  } : (tensor<5x7xf32>, tensor<f32>) -> tensor<5x7xf32>
  return %1 : tensor<5x7xf32>
  // CHECK: return %arg0 : tensor<5x7xf32>
}

// CHECK-LABEL: @pad_fold
func @pad_fold() -> tensor<4x5xi32> {
  %0 = constant dense<[[2, 3], [4, 5]]> : tensor<2x2xi32>
  %1 = constant dense<1> : tensor<i32>
  %3 = "mhlo.pad"(%0, %1) {
    edge_padding_low = dense<[1, 0]> : tensor<2xi64>,
    edge_padding_high = dense<[1, 2]> : tensor<2xi64>,
    interior_padding = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<2x2xi32>, tensor<i32>) -> tensor<4x5xi32>
  return %3 : tensor<4x5xi32>
  // CHECK: constant dense<[
  // CHECK-SAME: [1, 1, 1, 1, 1], [2, 1, 3, 1, 1], [4, 1, 5, 1, 1], [1, 1, 1, 1, 1]
  // CHECK-SAME: ]> : tensor<4x5xi32>
}

func @pad_fold_zero_elements() -> tensor<3xi32> {
  %0 = mhlo.constant dense<> : tensor<0xi32>
  %1 = mhlo.constant dense<7> : tensor<i32>
  %2 = "mhlo.pad"(%0, %1) {edge_padding_high = dense<3> : tensor<1xi64>, edge_padding_low = dense<0> : tensor<1xi64>, interior_padding = dense<0> : tensor<1xi64>} : (tensor<0xi32>, tensor<i32>) -> tensor<3xi32>
  return %2 : tensor<3xi32>
  // CHECK: mhlo.constant dense<7> : tensor<3xi32>
}

// CHECK-LABEL: @identity_broadcast_reshape
func @identity_broadcast_reshape(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %0 = "mhlo.broadcast"(%arg0) {
    broadcast_sizes = dense<[1]> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x128xf32>
  %1 = "mhlo.reshape"(%0) : (tensor<1x128xf32>) -> tensor<128xf32>
  return %1 : tensor<128xf32>
  // CHECK: return %arg0 : tensor<128xf32>
}

// CHECK-LABEL: @identity_broadcast_in_dim_reshape
func @identity_broadcast_in_dim_reshape(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = dense<[1]> : tensor<1xi64> } : (tensor<128xf32>) -> tensor<1x128xf32>
  %1 = "mhlo.reshape"(%0) : (tensor<1x128xf32>) -> tensor<128xf32>
  return %1 : tensor<128xf32>
  // CHECK: return %arg0 : tensor<128xf32>
}

// CHECK-LABEL: @broadcast_of_reshape
func @broadcast_of_reshape(%arg: tensor<?xf32>,
                           %shape: tensor<2xindex>) -> tensor<?x?xf32> {
  %0 = "mhlo.dynamic_reshape"(%arg, %shape)
      : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%0, %shape) {
    broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK: [[RESHAPE:%.*]] = "mhlo.dynamic_reshape"
// CHECK: return [[RESHAPE]]

// CHECK-LABEL: @permutation_broadcast_of_reshape
func @permutation_broadcast_of_reshape(%arg: tensor<?xf32>,
    %shape: tensor<2xindex>) -> tensor<?x?xf32> {
  %0 = "mhlo.dynamic_reshape"(%arg, %shape)
      : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%0, %shape) {
    broadcast_dimensions = dense<[1, 0]> : tensor<2xi64>
  } : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK: mhlo.dynamic_reshape
// CHECK: mhlo.dynamic_broadcast_in_dim

// CHECK-LABEL: @reshape_of_same_shape_op_result
func @reshape_of_same_shape_op_result(%arg: tensor<?xf32>,
    %shape: tensor<2xindex>) -> tensor<?x?xf32> {
  %0 = "mhlo.dynamic_reshape"(%arg, %shape)
    : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %1 = "mhlo.abs"(%0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "mhlo.dynamic_reshape"(%1, %shape)
    : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
// CHECK: mhlo.dynamic_reshape
// CHECK-NEXT: mhlo.abs
// CHECK-NOT: mhlo.dynamic_reshape
