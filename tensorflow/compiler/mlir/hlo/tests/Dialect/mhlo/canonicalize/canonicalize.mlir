// RUN: mlir-hlo-opt %s -pass-pipeline='func.func(canonicalize)' | FileCheck %s

// CHECK-LABEL: add_fold
func.func @add_fold() -> tensor<4xi64> {
  %0 = mhlo.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
  %1 = mhlo.constant dense<[5, 6, 7, 8]> : tensor<4xi64>
  // CHECK: mhlo.constant dense<[6, 8, 10, 12]>
  %2 = "mhlo.add"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  func.return %2 : tensor<4xi64>
}

// CHECK-LABEL: add_scalar_fold
func.func @add_scalar_fold() -> tensor<4xi64> {
  %0 = mhlo.constant dense<1> : tensor<4xi64>
  %1 = mhlo.constant dense<5> : tensor<4xi64>
  // CHECK: mhlo.constant dense<6>
  %2 = "mhlo.add"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  func.return %2 : tensor<4xi64>
}

// CHECK-LABEL: add_fold_float
func.func @add_fold_float() -> tensor<4xf64> {
  %0 = mhlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf64>
  %1 = mhlo.constant dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf64>
  // CHECK: mhlo.constant dense<[6.000000e+00, 8.000000e+00, 1.000000e+01, 1.200000e+01]>
  %2 = "mhlo.add"(%0, %1) : (tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>)
  func.return %2 : tensor<4xf64>
}

// CHECK-LABEL: add_zero_int_fold
func.func @add_zero_int_fold(%arg0: tensor<2x2xi64>) -> tensor<2x2xi64> {
  %0 = mhlo.constant dense<0> : tensor<2x2xi64>
  %1 = "mhlo.add"(%arg0, %0) : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
  // CHECK: return %arg0 : tensor<2x2xi64>
  func.return %1 : tensor<2x2xi64>
}

// CHECK-LABEL: add_zero_float_flod
func.func @add_zero_float_flod(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = mhlo.constant dense<0.0> : tensor<2x2xf32>
  %1 = "mhlo.add"(%0, %arg0) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: return %arg0 : tensor<2x2xf32>
  func.return %1 : tensor<2x2xf32>
}

// CHECK-LABEL: sub_scalar_fold
func.func @sub_scalar_fold() -> tensor<4xi64> {
  %0 = mhlo.constant dense<5> : tensor<4xi64>
  %1 = mhlo.constant dense<1> : tensor<4xi64>
  // CHECK: mhlo.constant dense<4>
  %2 = "mhlo.subtract"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  func.return %2 : tensor<4xi64>
}

// CHECK-LABEL: multiply_scalar_fold
func.func @multiply_scalar_fold() -> tensor<4xi64> {
  %0 = mhlo.constant dense<5> : tensor<4xi64>
  %1 = mhlo.constant dense<3> : tensor<4xi64>
  // CHECK: mhlo.constant dense<15>
  %2 = "mhlo.multiply"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  func.return %2 : tensor<4xi64>
}

// CHECK-LABEL: mul_one_int_fold
func.func @mul_one_int_fold(%arg0: tensor<2x2xi64>) -> tensor<2x2xi64> {
  %0 = mhlo.constant dense<1> : tensor<2x2xi64>
  %1 = "mhlo.multiply"(%arg0, %0) : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
  // CHECK: return %arg0 : tensor<2x2xi64>
  func.return %1 : tensor<2x2xi64>
}

// CHECK-LABEL: mul_one_int8_fold
func.func @mul_one_int8_fold(%arg0: tensor<2x2xi8>) -> tensor<2x2xi8> {
  %0 = mhlo.constant dense<1> : tensor<2x2xi8>
  %1 = "mhlo.multiply"(%arg0, %0) : (tensor<2x2xi8>, tensor<2x2xi8>) -> tensor<2x2xi8>
  // CHECK: return %arg0 : tensor<2x2xi8>
  func.return %1 : tensor<2x2xi8>
}

// CHECK-LABEL: mul_one_float_flod
func.func @mul_one_float_flod(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = mhlo.constant dense<1.0> : tensor<2x2xf32>
  %1 = "mhlo.multiply"(%0, %arg0) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: return %arg0 : tensor<2x2xf32>
  func.return %1 : tensor<2x2xf32>
}

// CHECK-LABEL: mul_one_fp16_flod
func.func @mul_one_fp16_flod(%arg0: tensor<2x2xf16>) -> tensor<2x2xf16> {
  %0 = mhlo.constant dense<1.0> : tensor<2x2xf16>
  %1 = "mhlo.multiply"(%0, %arg0) : (tensor<2x2xf16>, tensor<2x2xf16>) -> tensor<2x2xf16>
  // CHECK: return %arg0 : tensor<2x2xf16>
  func.return %1 : tensor<2x2xf16>
}

// CHECK-LABEL: mul_one_bf16_flod
func.func @mul_one_bf16_flod(%arg0: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
  %0 = mhlo.constant dense<1.0> : tensor<2x2xbf16>
  %1 = "mhlo.multiply"(%0, %arg0) : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
  // CHECK: return %arg0 : tensor<2x2xbf16>
  func.return %1 : tensor<2x2xbf16>
}


// CHECK-LABEL: divide_scalar_fold
func.func @divide_scalar_fold() -> tensor<4xi64> {
  %0 = mhlo.constant dense<7> : tensor<4xi64>
  %1 = mhlo.constant dense<5> : tensor<4xi64>
  // CHECK: mhlo.constant dense<1>
  %2 = "mhlo.divide"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  func.return %2 : tensor<4xi64>
}

// CHECK-LABEL: divide_scalar_fold_by_zero
func.func @divide_scalar_fold_by_zero() -> tensor<4xi64> {
  %0 = mhlo.constant dense<7> : tensor<4xi64>
  %1 = mhlo.constant dense<0> : tensor<4xi64>
  // CHECK: mhlo.divide
  %2 = "mhlo.divide"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  func.return %2 : tensor<4xi64>
}

// CHECK-LABEL: divide_fold_int
func.func @divide_fold_int() -> tensor<4xi32> {
  %0 = mhlo.constant dense<[1, -2, 3, 4]> : tensor<4xi32>
  %1 = mhlo.constant dense<[-1, -2, -3, 2]> : tensor<4xi32>
  // CHECK: %[[RESULT:.+]] = mhlo.constant dense<[-1, 1, -1, 2]>
  %2 = "mhlo.divide"(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> (tensor<4xi32>)
  // CHECK: return %[[RESULT]]
  func.return %2 : tensor<4xi32>
}

// CHECK-LABEL: divide_fold_unsigned
func.func @divide_fold_unsigned() -> tensor<4xui32> {
  %0 = mhlo.constant dense<[1, -2, 3, 4]> : tensor<4xi32>
  %1 = "mhlo.convert"(%0) : (tensor<4xi32>) -> tensor<4xui32>
  %2 = mhlo.constant dense<[-1, -2, -3, 2]> : tensor<4xi32>
  %3 = "mhlo.convert"(%2) : (tensor<4xi32>) -> tensor<4xui32>
  // CHECK: %[[RESULT:.+]] = mhlo.constant dense<[0, 1, 0, 2]>
  %4 = "mhlo.divide"(%1, %3) : (tensor<4xui32>, tensor<4xui32>) -> (tensor<4xui32>)
  // CHECK: return %[[RESULT]]
  func.return %4 : tensor<4xui32>
}

// CHECK-LABEL: divide_fold_float
func.func @divide_fold_float() -> tensor<4xf64> {
  %0 = mhlo.constant dense<[5.0, 66.0, 5.0, 1.0]> : tensor<4xf64>
  %1 = mhlo.constant dense<[5.0, 3.0, 2.0, 4.0]> : tensor<4xf64>
  // CHECK: mhlo.constant dense<[1.000000e+00, 2.200000e+01, 2.500000e+00, 2.500000e-01]>
  %2 = "mhlo.divide"(%0, %1) : (tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>)
  func.return %2 : tensor<4xf64>
}

// CHECK-LABEL: divide_fold_by_zero
func.func @divide_fold_by_zero() -> tensor<4xi64> {
  %0 = mhlo.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
  %1 = mhlo.constant dense<[1, 2, 3, 0]> : tensor<4xi64>
  // CHECK: mhlo.divide
  %2 = "mhlo.divide"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  func.return %2 : tensor<4xi64>
}

// CHECK-LABEL: remainder_scalar_fold_by_zero
func.func @remainder_scalar_fold_by_zero() -> tensor<4xi64> {
  %0 = mhlo.constant dense<7> : tensor<4xi64>
  %1 = mhlo.constant dense<0> : tensor<4xi64>
  // CHECK: mhlo.remainder
  %2 = "mhlo.remainder"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  func.return %2 : tensor<4xi64>
}

// CHECK-LABEL: remainder_fold_int
func.func @remainder_fold_int() -> tensor<4xi32> {
  %0 = mhlo.constant dense<[5, 66, 5, -1]> : tensor<4xi32>
  %1 = mhlo.constant dense<[3, 5, 1, -2]> : tensor<4xi32>
  // CHECK: %[[RESULT:.+]] = mhlo.constant dense<[2, 1, 0, -1]>
  %2 = "mhlo.remainder"(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> (tensor<4xi32>)
  // CHECK: return %[[RESULT]]
  func.return %2 : tensor<4xi32>
}

// CHECK-LABEL: remainder_fold_float
func.func @remainder_fold_float() -> tensor<4xf32> {
  %0 = mhlo.constant dense<[7.0, 66.5, 5.0, 3.1]> : tensor<4xf32>
  %1 = mhlo.constant dense<[3.0, 5.0, 1.0, 2.6]> : tensor<4xf32>
  // CHECK: mhlo.constant dense<[1.000000e+00, 1.500000e+00, 0.000000e+00, 5.000000e-01]>
  %2 = "mhlo.remainder"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)
  func.return %2 : tensor<4xf32>
}

// CHECK-LABEL: round_fold
func.func @round_fold() -> tensor<4xf32> {
  %0 = mhlo.constant dense<[-1.5, -0.1, 1.1, 2.5]> : tensor<4xf32>
  %1 = "mhlo.round_nearest_afz"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>
  // CHECK: mhlo.constant dense<[-2.000000e+00, -0.000000e+00, 1.000000e+00, 3.000000e+00]>
}

// CHECK-LABEL: max_scalar_fold
func.func @max_scalar_fold() -> tensor<4xi64> {
  %0 = mhlo.constant dense<7> : tensor<4xi64>
  %1 = mhlo.constant dense<-5> : tensor<4xi64>
  // CHECK: %[[RESULT:.+]] = mhlo.constant dense<7>
  %2 = "mhlo.maximum"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  // CHECK: return %[[RESULT]]
  func.return %2 : tensor<4xi64>
}

// CHECK-LABEL: max_scalar_fold_unsigned
func.func @max_scalar_fold_unsigned() -> tensor<4xui32> {
  %0 = mhlo.constant dense<7> : tensor<4xui32>
  %1 = mhlo.constant dense<-5> : tensor<4xi32>
  %2 = "mhlo.convert"(%1) : (tensor<4xi32>) -> tensor<4xui32>
  // CHECK: %[[RESULT:.+]] = mhlo.constant dense<4294967291>
  %3 = "mhlo.maximum"(%0, %2) : (tensor<4xui32>, tensor<4xui32>) -> (tensor<4xui32>)
  // CHECK: return %[[RESULT]]
  func.return %3 : tensor<4xui32>
}

// CHECK-LABEL: max_fold_float
func.func @max_fold_float() -> tensor<4xf64> {
  %0 = mhlo.constant dense<[5.0, 66.0, 5.0, 1.0]> : tensor<4xf64>
  %1 = mhlo.constant dense<[5.0, 3.0, 2.0, 4.0]> : tensor<4xf64>
  // CHECK: mhlo.constant dense<[5.000000e+00, 6.600000e+01, 5.000000e+00, 4.000000e+00]>
  %2 = "mhlo.maximum"(%0, %1) : (tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>)
  func.return %2 : tensor<4xf64>
}

// CHECK-LABEL: min_scalar_fold
func.func @min_scalar_fold() -> tensor<4xi64> {
  %0 = mhlo.constant dense<7> : tensor<4xi64>
  %1 = mhlo.constant dense<-5> : tensor<4xi64>
  // CHECK: mhlo.constant dense<-5>
  %2 = "mhlo.minimum"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>)
  func.return %2 : tensor<4xi64>
}

// CHECK-LABEL: min_fold_float
func.func @min_fold_float() -> tensor<4xf64> {
  %0 = mhlo.constant dense<[5.0, 66.0, 5.0, 1.0]> : tensor<4xf64>
  %1 = mhlo.constant dense<[5.0, 3.0, 2.0, 4.0]> : tensor<4xf64>
  // CHECK: mhlo.constant dense<[5.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00]>
  %2 = "mhlo.minimum"(%0, %1) : (tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>)
  func.return %2 : tensor<4xf64>
}

// CHECK-LABEL: concatenate_noop
func.func @concatenate_noop(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK-SAME: [[ARG:%.+]]: tensor<4xi32>
  %0 = "mhlo.concatenate"(%arg0) { dimension = 0 : i64 } : (tensor<4xi32>) -> tensor<4xi32>

  // CHECK: return [[ARG]]
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: concatenate_noop_typecast
func.func @concatenate_noop_typecast(%arg0: tensor<?xi32>) -> tensor<4xi32> {
  // CHECK-SAME: [[ARG:%.+]]: tensor<?xi32>
  // CHECK-NEXT: [[RES:%.+]] = tensor.cast [[ARG]] : tensor<?xi32> to tensor<4xi32>
  %0 = "mhlo.concatenate"(%arg0) { dimension = 0 : i64 } : (tensor<?xi32>) -> tensor<4xi32>

  // CHECK: return [[RES]]
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: concatenate_remove_operand
func.func @concatenate_remove_operand(%arg0: tensor<4xi32>, %arg1: tensor<0xi32>) -> tensor<4xi32> {
  // CHECK-SAME: [[ARG0:%.+]]: tensor<4xi32>
  // CHECK-SAME: [[ARG1:%.+]]: tensor<0xi32>
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<4xi32>, tensor<0xi32>) -> tensor<4xi32>

  // CHECK: return [[ARG0]]
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: concatenate_forward
func.func @concatenate_forward(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<12xi32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<4xi32>, tensor<4xi32>) -> tensor<8xi32>
  %1 = mhlo.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  // CHECK: "mhlo.concatenate"(%arg0, %arg1, %0) {dimension = 0 : i64} : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<12xi32>
  %2 = "mhlo.concatenate"(%0, %1) { dimension = 0 : i64 } : (tensor<8xi32>, tensor<4xi32>) -> tensor<12xi32>

  func.return %2 : tensor<12xi32>
}

// CHECK-LABEL: concatenate_empty_bool
func.func @concatenate_empty_bool(%arg0: tensor<0xi1>, %arg1: tensor<0xi1>) -> tensor<0xi1> {
  // CHECK: mhlo.constant
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<0xi1>, tensor<0xi1>) -> tensor<0xi1>

  func.return %0 : tensor<0xi1>
}

// CHECK-LABEL: concatenate_empty_int
func.func @concatenate_empty_int(%arg0: tensor<0xi32>, %arg1: tensor<0xi32>) -> tensor<0xi32> {
  // CHECK: mhlo.constant
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<0xi32>, tensor<0xi32>) -> tensor<0xi32>

  func.return %0 : tensor<0xi32>
}

// CHECK-LABEL: concatenate_empty_float
func.func @concatenate_empty_float(%arg0: tensor<0xf32>, %arg1: tensor<0xf32>) -> tensor<0xf32> {
  // CHECK: mhlo.constant
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<0xf32>, tensor<0xf32>) -> tensor<0xf32>

  func.return %0 : tensor<0xf32>
}

// CHECK-LABEL: concatenate_const_1D
func.func @concatenate_const_1D() -> tensor<4xi32> {
  // CHECK: [[VAL:%.+]]= mhlo.constant dense<[0, 1, 2, 3]>
  %0 = mhlo.constant dense<[0, 1]> : tensor<2xi32>
  %1 = mhlo.constant dense<[2, 3]> : tensor<2xi32>
  %2 = "mhlo.concatenate"(%0, %1) { dimension = 0 : i64 } : (tensor<2xi32>, tensor<2xi32>) -> tensor<4xi32>

  // CHECK: return [[VAL]]
  func.return %2 : tensor<4xi32>
}

// CHECK-LABEL: concatenate_const_1D_float
func.func @concatenate_const_1D_float() -> tensor<4xf32> {
  // CHECK: [[VAL:%.+]] = mhlo.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]>

  %0 = mhlo.constant dense<[0.0, 1.0]> : tensor<2xf32>
  %1 = mhlo.constant dense<[2.0, 3.0]> : tensor<2xf32>
  %2 = "mhlo.concatenate"(%0, %1) { dimension = 0 : i64 } : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>

  // CHECK: return [[VAL]]
  func.return %2 : tensor<4xf32>
}

// CHECK-LABEL: concatenate_const_2D_vertical
func.func @concatenate_const_2D_vertical() -> tensor<2x2xi32> {
  // CHECK: [[VAL:%.+]]= mhlo.constant dense<[
  // CHECK-SAME: [0, 1], [2, 3]
  // CHECK-SAME: ]>
  %0 = mhlo.constant dense<[[0, 1]]> : tensor<1x2xi32>
  %1 = mhlo.constant dense<[[2, 3]]> : tensor<1x2xi32>
  %2 = "mhlo.concatenate"(%0, %1) { dimension = 0 : i64 } : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>

  // CHECK: return [[VAL]]
  func.return %2 : tensor<2x2xi32>
}

// CHECK-LABEL: concatenate_const_2D_horizontal
func.func @concatenate_const_2D_horizontal() -> tensor<2x2xi32> {
  // CHECK: [[VAL:%.+]]= mhlo.constant dense<[
  // CHECK-SAME: [0, 2], [1, 3]
  // CHECK-SAME: ]>
  %0 = mhlo.constant dense<[[0], [1]]> : tensor<2x1xi32>
  %1 = mhlo.constant dense<[[2], [3]]> : tensor<2x1xi32>
  %2 = "mhlo.concatenate"(%0, %1) { dimension = 1 : i64 } : (tensor<2x1xi32>, tensor<2x1xi32>) -> tensor<2x2xi32>

  // CHECK: return [[VAL]]
  func.return %2 : tensor<2x2xi32>
}

// CHECK-LABEL: constant_like_constant
func.func @constant_like_constant(%arg0: tensor<3x4xi32>) -> tensor<3x4xf32> {
  // CHECK: mhlo.constant dense<3.200000e+00>
  %0 = "chlo.constant_like"(%arg0) { value = 3.2 : f32 } : (tensor<3x4xi32>) -> tensor<3x4xf32>
  func.return %0 : tensor<3x4xf32>
}

// CHECK-LABEL: constant_like_constant_dynamic
func.func @constant_like_constant_dynamic(%arg0: tensor<*xi32>) -> tensor<*xf32> {
  // CHECK: chlo.constant_like
  %0 = "chlo.constant_like"(%arg0) { value = 3.2 : f32 } : (tensor<*xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: dynamic_update_slice_identity_update
func.func @dynamic_update_slice_identity_update(%arg0: tensor<3x4xi64>, %arg1: tensor<3x4xi64>) -> tensor<3x4xi64> {
  // CHECK: return %arg1
  %0 = mhlo.constant dense<0> : tensor<i64>
  %1 = "mhlo.dynamic_update_slice"(%arg0, %arg1, %0, %0) : (tensor<3x4xi64>, tensor<3x4xi64>, tensor<i64>, tensor<i64>) -> tensor<3x4xi64>
  func.return %1 : tensor<3x4xi64>
}

// CHECK-LABEL: dynamic_update_slice_fold_fail_dynamic_shapes
func.func @dynamic_update_slice_fold_fail_dynamic_shapes(%arg0: tensor<?x?xi64>, %arg1: tensor<?x?xi64>) -> tensor<?x?xi64> {
  %0 = mhlo.constant dense<0> : tensor<i64>
  %1 = "mhlo.dynamic_update_slice"(%arg0, %arg1, %0, %0) : (tensor<?x?xi64>, tensor<?x?xi64>, tensor<i64>, tensor<i64>) -> tensor<?x?xi64>
  func.return %1 : tensor<?x?xi64>
  // CHECK: %[[CST:.*]] = mhlo.constant dense<0> : tensor<i64>
  // CHECK: %[[VAL:.*]] = "mhlo.dynamic_update_slice"(%arg0, %arg1, %[[CST]], %[[CST]]) : (tensor<?x?xi64>, tensor<?x?xi64>, tensor<i64>, tensor<i64>) -> tensor<?x?xi64>
  // CHECK: return %[[VAL]] : tensor<?x?xi64>
}

// CHECK-LABEL: dynamic_slice_variable_start
func.func @dynamic_slice_variable_start(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  // CHECK: "mhlo.dynamic_slice"
  %1 = "mhlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %1 : tensor<1x4xi32>
}

// CHECK-LABEL: dynamic_slice_constant_start
func.func @dynamic_slice_constant_start(%arg0: tensor<4xi32>) -> tensor<2xi32> {
  // CHECK: %[[RESULT:.*]] =  "mhlo.slice"(%arg0)
  // CHECK-DAG-SAME: limit_indices = dense<3> : tensor<1xi64>
  // CHECK-DAG-SAME: start_indices = dense<1> : tensor<1xi64>
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>}
  // CHECK: return %[[RESULT]] : tensor<2xi32>
  %0 = mhlo.constant dense<1> : tensor<i64>
  %1 = "mhlo.dynamic_slice"(%arg0, %0) {slice_sizes = dense<2> : tensor<1xi64>} : (tensor<4xi32>, tensor<i64>) -> tensor<2xi32>
  func.return %1 : tensor<2xi32>
}

// CHECK-LABEL: dynamic_slice_constant_start_dynamic_shape
func.func @dynamic_slice_constant_start_dynamic_shape(%arg0: tensor<?x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  // CHECK: mhlo.dynamic_slice
  // CHECK-NOT: mhlo.slice
  %0 = mhlo.constant dense<1> : tensor<i64>
  %1 = mhlo.constant dense<0> : tensor<i64>
  %2 = "mhlo.dynamic_slice"(%arg0, %0, %1) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<?x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %2 : tensor<1x4xi32>
}

// CHECK-LABEL: dynamic_slice_constant_start_upper_bound
func.func @dynamic_slice_constant_start_upper_bound(%arg0: tensor<8x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  // CHECK: %[[RESULT:.*]] = "mhlo.slice"(%arg0)
  // CHECK-SAME: limit_indices = dense<[8, 4]> : tensor<2xi64>
  // CHECK-SAME: start_indices = dense<[7, 0]> : tensor<2xi64>
  // CHECK-SAME: strides = dense<1> : tensor<2xi64>
  // CHECK: return %[[RESULT]] : tensor<1x4xi32>
  %0 = mhlo.constant dense<10> : tensor<i64>
  %1 = mhlo.constant dense<0> : tensor<i64>
  %2 = "mhlo.dynamic_slice"(%arg0, %0, %1) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<8x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %2 : tensor<1x4xi32>
}

// CHECK-LABEL: dynamic_slice_constant_start_lower_bound
func.func @dynamic_slice_constant_start_lower_bound(%arg0: tensor<8x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  // CHECK: %[[RESULT:.*]] = "mhlo.slice"(%arg0)
  // CHECK-SAME: limit_indices = dense<[1, 4]> : tensor<2xi64>
  // CHECK-SAME: start_indices = dense<0> : tensor<2xi64>
  // CHECK-SAME: strides = dense<1> : tensor<2xi64>
  // CHECK: return %[[RESULT]] : tensor<1x4xi32>
  %0 = mhlo.constant dense<-1> : tensor<i64>
  %1 = mhlo.constant dense<0> : tensor<i64>
  %2 = "mhlo.dynamic_slice"(%arg0, %0, %1) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<8x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %2 : tensor<1x4xi32>
}

// CHECK-LABEL: slice_2D_noop
// CHECK-SAME: [[ARG:%.+]]: tensor<2x2xi64>
func.func @slice_2D_noop(%arg0: tensor<2x2xi64>) -> tensor<2x2xi64> {
  %0 = "mhlo.slice"(%arg0) { limit_indices = dense<[2, 2]> : tensor<2xi64>, start_indices = dense<[0, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x2xi64>) -> (tensor<2x2xi64>)

  // CHECK-NEXT: return [[ARG]]
  func.return %0 : tensor<2x2xi64>
}

// CHECK-LABEL: slice_1D_fold
func.func @slice_1D_fold() -> tensor<2xi64> {
  %0 = mhlo.constant dense<[5, 7, 9, 10]> : tensor<4xi64>
  // CHECK: mhlo.constant dense<[7, 9]>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[3]> : tensor<1xi64>, start_indices = dense<[1]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xi64>) -> (tensor<2xi64>)
  func.return %1 : tensor<2xi64>
}

// CHECK-LABEL: slice_1D_fp
func.func @slice_1D_fp() -> tensor<2xf32> {
  %0 = mhlo.constant dense<[5.0, 7.0, 9.0, 10.0]> : tensor<4xf32>
  // CHECK: mhlo.constant dense<[7.000000e+00, 9.000000e+00]>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[3]> : tensor<1xi64>, start_indices = dense<[1]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> (tensor<2xf32>)
  func.return %1 : tensor<2xf32>
}

// CHECK-LABEL: slice_1D_strided_fold
func.func @slice_1D_strided_fold() -> tensor<2xi64> {
  %0 = mhlo.constant dense<[5, 7, 9, 10]> : tensor<4xi64>
  // CHECK: mhlo.constant dense<[7, 10]>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[4]> : tensor<1xi64>, start_indices = dense<[1]> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>} : (tensor<4xi64>) -> (tensor<2xi64>)
  func.return %1 : tensor<2xi64>
}

// CHECK-LABEL: slice_2D_fold
func.func @slice_2D_fold() -> tensor<2x2xi64> {
  %0 = mhlo.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi64>
  // CHECK-NEXT: mhlo.constant dense<[
  // CHECK-SAME: [6, 7],
  // CHECK-SAME: [10, 11]
  // CHECK-SAME: ]>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[3, 4]> : tensor<2xi64>, start_indices = dense<[1, 2]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x4xi64>) -> (tensor<2x2xi64>)
  func.return %1 : tensor<2x2xi64>
}

// CHECK-LABEL: slice_2D_fold_horizontal
func.func @slice_2D_fold_horizontal() -> tensor<1x4xi64> {
  %0 = mhlo.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi64>
  // CHECK-NEXT: mhlo.constant dense<[
  // CHECK-SAME: [0, 1, 2, 3]
  // CHECK-SAME: ]>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[1, 4]> : tensor<2xi64>, start_indices = dense<[0, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x4xi64>) -> (tensor<1x4xi64>)
  func.return %1 : tensor<1x4xi64>
}

// CHECK-LABEL: slice_2D_fold_vertical
func.func @slice_2D_fold_vertical() -> tensor<4x1xi64> {
  %0 = mhlo.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : tensor<4x4xi64>
  // CHECK-NEXT: mhlo.constant dense<[
  // CHECK-SAME: [2], [6], [10], [14]
  // CHECK-SAME: ]>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[4, 3]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x4xi64>) -> (tensor<4x1xi64>)
  func.return %1 : tensor<4x1xi64>
}

// CHECK-LABEL: slice_zero_elements
func.func @slice_zero_elements() -> tensor<0xi64> {
  %0 = mhlo.constant dense<> : tensor<0xi64>
  // CHECK: %[[CONST:.*]] = mhlo.constant dense<> : tensor<0xi64>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[0]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<0xi64>) -> (tensor<0xi64>)
  // CHECK: return %[[CONST]] : tensor<0xi64>
  func.return %1 : tensor<0xi64>
}

// CHECK-LABEL: slice_unknown_shape
func.func @slice_unknown_shape(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "mhlo.slice"(%arg0) {limit_indices = dense<[1, 4]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[1, 4]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: slice_concat_fold_first
func.func @slice_concat_fold_first(%arg0: tensor<1x5xf32>, %arg1: tensor<1x5xf32>) -> tensor<1x5xf32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<2x5xf32>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[1, 5]> : tensor<2xi64>, start_indices = dense<[0, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x5xf32>) -> (tensor<1x5xf32>)
  // CHECK: return %arg0
  func.return %1 : tensor<1x5xf32>
}

// CHECK-LABEL: slice_concat_fold_second
func.func @slice_concat_fold_second(%arg0: tensor<1x5xf32>, %arg1: tensor<1x5xf32>) -> tensor<1x5xf32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<2x5xf32>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[2, 5]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x5xf32>) -> (tensor<1x5xf32>)
  // CHECK: return %arg1
  func.return %1 : tensor<1x5xf32>
}

// CHECK-LABEL: slice_concat_fold_second_with_slice
func.func @slice_concat_fold_second_with_slice(%arg0: tensor<1x5xf32>, %arg1: tensor<1x5xf32>) -> tensor<1x4xf32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<2x5xf32>
  // CHECK: [[SLICE:%.+]] = "mhlo.slice"(%arg1) {limit_indices = dense<[1, 5]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x5xf32>) -> tensor<1x4xf32>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[2, 5]> : tensor<2xi64>, start_indices = dense<[1, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x5xf32>) -> (tensor<1x4xf32>)

  // CHECK: return [[SLICE]]
  func.return %1 : tensor<1x4xf32>
}

// CHECK-LABEL: slice_concat_fold_middle
func.func @slice_concat_fold_middle(%arg0: tensor<1x5xf32>, %arg1: tensor<2x5xf32>, %arg2: tensor<1x5xf32>) -> tensor<1x5xf32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1, %arg2) { dimension = 0 : i64 } : (tensor<1x5xf32>, tensor<2x5xf32>, tensor<1x5xf32>) -> tensor<4x5xf32>
  // CHECK: [[SLICE:%.+]] = "mhlo.slice"(%arg1) {limit_indices = dense<[2, 5]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[3, 5]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x5xf32>) -> (tensor<1x5xf32>)

  // CHECK: return [[SLICE]]
  func.return %1 : tensor<1x5xf32>
}

// CHECK-LABEL: slice_concat_fold_two
func.func @slice_concat_fold_two(%arg0: tensor<1x5xf32>, %arg1: tensor<2x5xf32>, %arg2: tensor<1x5xf32>) -> tensor<2x5xf32> {
  // CHECK: [[CONCAT:%.+]] = "mhlo.concatenate"(%arg1, %arg2) {dimension = 0 : i64}
  %0 = "mhlo.concatenate"(%arg0, %arg1, %arg2) { dimension = 0 : i64 } : (tensor<1x5xf32>, tensor<2x5xf32>, tensor<1x5xf32>) -> tensor<4x5xf32>

  // CHECK: [[SLICE:%.+]] = "mhlo.slice"([[CONCAT]]) {limit_indices = dense<[3, 5]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[4, 5]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x5xf32>) -> (tensor<2x5xf32>)

  // CHECK: return [[SLICE]]
  func.return %1 : tensor<2x5xf32>
}

// CHECK-LABEL: slice_concat_empty
func.func @slice_concat_empty(%arg0: tensor<1x5xf32>, %arg1: tensor<1x5xf32>, %arg2: tensor<1x5xf32>) -> tensor<1x5xf32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<2x5xf32>
  %1 = "mhlo.slice"(%0) { limit_indices = dense<[2, 5]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x5xf32>) -> (tensor<0x5xf32>)
  %2 = "mhlo.concatenate"(%1, %arg2) { dimension = 0 : i64 } : (tensor<0x5xf32>, tensor<1x5xf32>) -> tensor<1x5xf32>

  // CHECK: return %arg2
  func.return %2 : tensor<1x5xf32>
}

// CHECK-LABEL: func @broadcast_identity
func.func @broadcast_identity(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  // CHECK: return %arg0
  %0 = "mhlo.broadcast"(%arg0) {broadcast_sizes = dense<[]> : tensor<0xi64>} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  func.return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func @broadcast_dynamic_shape_identity
func.func @broadcast_dynamic_shape_identity(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK: return %arg0
  %0 = "mhlo.broadcast"(%arg0) {broadcast_sizes = dense<[]> : tensor<0xi64>} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: func @broadcast_dynamic_shape_not_identity
func.func @broadcast_dynamic_shape_not_identity(%arg0: tensor<?x?x?xf32>) -> tensor<20x?x?x?xf32> {
  // CHECK: mhlo.broadcast
  %0 = "mhlo.broadcast"(%arg0) {broadcast_sizes = dense<[20]> : tensor<1xi64>} : (tensor<?x?x?xf32>) -> tensor<20x?x?x?xf32>
  func.return %0 : tensor<20x?x?x?xf32>
}

// CHECK-LABEL: func @broadcast_constant_fold_0d
func.func @broadcast_constant_fold_0d() -> tensor<1x64x224x224xf32> {
  %cst = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %b = "mhlo.broadcast"(%cst) {broadcast_sizes = dense<[1, 64, 224, 224]> : tensor<4xi64>} : (tensor<f32>) -> tensor<1x64x224x224xf32>
  func.return %b : tensor<1x64x224x224xf32>
}
// CHECK-NEXT: %[[CST:.*]] = mhlo.constant dense<0.000000e+00> : tensor<1x64x224x224xf32>
// CHECK-NEXT: return %[[CST]] : tensor<1x64x224x224xf32>

// CHECK-LABEL: func @broadcast_constant_fold
func.func @broadcast_constant_fold() -> tensor<1x64x4x4xf32> {
  %cst = mhlo.constant dense<0.000000e+00> : tensor<4x4xf32>
  %b = "mhlo.broadcast"(%cst) {broadcast_sizes = dense<[1, 64]> : tensor<2xi64>} : (tensor<4x4xf32>) -> tensor<1x64x4x4xf32>
  func.return %b : tensor<1x64x4x4xf32>
}
// CHECK-NEXT: %[[CST:.*]] = mhlo.constant dense<0.000000e+00> : tensor<1x64x4x4xf32>
// CHECK-NEXT: return %[[CST]] : tensor<1x64x4x4xf32>

// CHECK-LABEL: func @broadcast_constant_fold_not_splat
func.func @broadcast_constant_fold_not_splat() -> tensor<1x64x2xf32> {
  // CHECK: mhlo.constant
  %cst = mhlo.constant dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf32>
  // CHECK: mhlo.broadcast
  %b = "mhlo.broadcast"(%cst) {broadcast_sizes = dense<[1, 64]> : tensor<2xi64>} : (tensor<2xf32>) -> tensor<1x64x2xf32>
  func.return %b : tensor<1x64x2xf32>
}

// CHECK-LABEL: func @broadcast_constant_fold_complex
func.func @broadcast_constant_fold_complex() -> tensor<1x64x224x224xcomplex<f32>> {
  %cst = mhlo.constant dense<(0.000000e+00,1.000000e+00)> : tensor<complex<f32>>
  %b = "mhlo.broadcast"(%cst) {broadcast_sizes = dense<[1, 64, 224, 224]> : tensor<4xi64>} : (tensor<complex<f32>>) -> tensor<1x64x224x224xcomplex<f32>>
  func.return %b : tensor<1x64x224x224xcomplex<f32>>
}
// CHECK-NEXT: %[[CST:.*]] = mhlo.constant dense<(0.000000e+00,1.000000e+00)> : tensor<1x64x224x224xcomplex<f32>>
// CHECK-NEXT: return %[[CST]] : tensor<1x64x224x224xcomplex<f32>>

// CHECK-LABEL: func @broadcast_in_dim_identity
func.func @broadcast_in_dim_identity(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  // CHECK: return %arg0
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  func.return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func @broadcast_in_dim_equivalent_reshape
func.func @broadcast_in_dim_equivalent_reshape(%arg0: tensor<2x3x4xf32>) -> tensor<1x2x3x4xf32> {
  // CHECK: mhlo.reshape
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 2, 3]> : tensor<3xi64>} : (tensor<2x3x4xf32>) -> tensor<1x2x3x4xf32>
  func.return %0 : tensor<1x2x3x4xf32>
}

// CHECK-LABEL: func @broadcast_in_dim_not_identity_because_it_actually_broadcasts
func.func @broadcast_in_dim_not_identity_because_it_actually_broadcasts(%arg0: tensor<1x2xf32>) -> tensor<2x2xf32> {
  // CHECK: mhlo.broadcast_in_dim
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func @broadcast_in_dim_equivalent_transpose
func.func @broadcast_in_dim_equivalent_transpose(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: mhlo.transpose
  // CHECK-SAME: permutation = dense<[1, 0]>
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 0]> : tensor<2xi64>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func @broadcast_consecutive
func.func @broadcast_consecutive(%arg0: tensor<2x3xf32>) -> tensor<2x3x4x5xf32> {
  // CHECK: mhlo.broadcast_in_dim
  // CHECK-SAME: broadcast_dimensions = dense<[0, 1]>
  // CHECK-NEXT: return
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x3xf32>) -> tensor<2x3x4xf32>
  %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<2x3x4xf32>) -> tensor<2x3x4x5xf32>
  func.return %1 : tensor<2x3x4x5xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic
func.func @dynamic_broadcast_in_dim_op_not_actually_dynamic(%arg0: tensor<4xf32>, %arg1: tensor<2xi64>) -> tensor<5x4xf32> {
  // CHECK: %[[RESULT:.+]] = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<5x4xf32>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %arg1) { broadcast_dimensions = dense<1> : tensor<1xi64> } : (tensor<4xf32>, tensor<2xi64>) -> tensor<5x4xf32>
  // CHECK: return %[[RESULT]] : tensor<5x4xf32>
  func.return %0 : tensor<5x4xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_shape
func.func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_shape(%arg0: tensor<i32>) -> tensor<4x32xi32> {
  %0 = mhlo.constant dense<[4, 32]> : tensor<2xi32>
  // CHECK: %[[RESULT:.+]] = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i32>) -> tensor<4x32xi32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<i32>, tensor<2xi32>) -> tensor<?x32xi32>
  %2 = "mhlo.dynamic_reshape"(%1, %0) : (tensor<?x32xi32>, tensor<2xi32>) -> tensor<4x32xi32>
  // CHECK: return %[[RESULT]] : tensor<4x32xi32>
  func.return %2 : tensor<4x32xi32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_index_shape
func.func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_index_shape(%arg0: tensor<f32>) -> tensor<4x32xf32> {
  %0 = shape.const_shape [4, 32] : tensor<2xindex>
  // CHECK: %[[RESULT:.+]] = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<4x32xf32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x32xf32>
  %2 = "mhlo.dynamic_reshape"(%1, %0) : (tensor<?x32xf32>, tensor<2xindex>) -> tensor<4x32xf32>
  // CHECK: return %[[RESULT]] : tensor<4x32xf32>
  func.return %2 : tensor<4x32xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_requires_cast
func.func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_requires_cast(%arg0: tensor<f32>) -> tensor<?x?xf32> {
  %0 = shape.const_shape [4, 32] : tensor<2xindex>
  // CHECK: %[[BCAST:.+]] = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<4x32xf32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK: %[[RESULT:.*]] = tensor.cast %[[BCAST]] : tensor<4x32xf32> to tensor<?x?xf32>
  // CHECK: return %[[RESULT]] : tensor<?x?xf32>
  func.return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_almost_not_actually_dynamic
func.func @dynamic_broadcast_in_dim_op_almost_not_actually_dynamic(%arg0: tensor<?xf32>, %arg1: tensor<2xi64>) -> tensor<5x4xf32> {
  // CHECK: %[[RESULT:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<2xi64>) -> tensor<5x4xf32>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %arg1) { broadcast_dimensions = dense<1> : tensor<1xi64> } : (tensor<?xf32>, tensor<2xi64>) -> tensor<5x4xf32>
  // CHECK: return %[[RESULT]] : tensor<5x4xf32>
  func.return %0 : tensor<5x4xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_to_same_shape_1
func.func @dynamic_broadcast_in_dim_to_same_shape_1(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<?xf32>
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<1xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %0) { broadcast_dimensions = dense<0> : tensor<1xi64> } : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK: return %[[ARG]] : tensor<?xf32>
  func.return %2 : tensor<?xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_to_same_shape_2
func.func @dynamic_broadcast_in_dim_to_same_shape_2(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<?xf32>
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> !shape.shape
  %1 = shape.to_extent_tensor %0 : !shape.shape -> tensor<1xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %1) { broadcast_dimensions = dense<0> : tensor<1xi64> } : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK: return %[[ARG]] : tensor<?xf32>
  func.return %2 : tensor<?xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_to_same_shape_3
func.func @dynamic_broadcast_in_dim_to_same_shape_3(%arg0: tensor<*xf32>) -> tensor<?xf32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<*xf32>
  %0 = shape.shape_of %arg0 : tensor<*xf32> -> tensor<?xindex>
  %1 = tensor.cast %0 : tensor<?xindex> to tensor<1xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %1) { broadcast_dimensions = dense<0> : tensor<1xi64> } : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK: %[[RES:.*]] = tensor.cast %[[ARG]] : tensor<*xf32> to tensor<?xf32>
  // CHECK: return %[[RES]] : tensor<?xf32>
  func.return %2 : tensor<?xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_to_same_shape_4
func.func @dynamic_broadcast_in_dim_to_same_shape_4(%arg0: tensor<*xf32>) -> tensor<?xf32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<*xf32>
  %0 = shape.shape_of %arg0 : tensor<*xf32> -> !shape.shape
  %1 = shape.to_extent_tensor %0 : !shape.shape -> tensor<?xindex>
  %2 = tensor.cast %1 : tensor<?xindex> to tensor<1xindex>
  %3 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %2) { broadcast_dimensions = dense<0> : tensor<1xi64> } : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK: %[[RES:.*]] = tensor.cast %[[ARG]] : tensor<*xf32> to tensor<?xf32>
  // CHECK: return %[[RES]] : tensor<?xf32>
  func.return %3 : tensor<?xf32>
}

// CHECK-LABEL: func @broadcast_in_dim_constant_fold_0d
func.func @broadcast_in_dim_constant_fold_0d() -> tensor<1x64x224x224xf32> {
  %cst = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %b = "mhlo.broadcast_in_dim"(%cst) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<f32>) -> tensor<1x64x224x224xf32>
  func.return %b : tensor<1x64x224x224xf32>
}
// CHECK-NEXT: %[[CST:.*]] = mhlo.constant dense<0.000000e+00> : tensor<1x64x224x224xf32>
// CHECK-NEXT: return %[[CST]] : tensor<1x64x224x224xf32>

// CHECK-LABEL: func @broadcast_in_dim_constant_fold
func.func @broadcast_in_dim_constant_fold() -> tensor<1x64x4x4xf32> {
  %cst = mhlo.constant dense<0.000000e+00> : tensor<4x4xf32>
  %b = "mhlo.broadcast_in_dim"(%cst) {broadcast_dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<4x4xf32>) -> tensor<1x64x4x4xf32>
  func.return %b : tensor<1x64x4x4xf32>
}
// CHECK-NEXT: %[[CST:.*]] = mhlo.constant dense<0.000000e+00> : tensor<1x64x4x4xf32>
// CHECK-NEXT: return %[[CST]] : tensor<1x64x4x4xf32>

// CHECK-LABEL: func @broadcast_in_dim_constant_fold_complex
func.func @broadcast_in_dim_constant_fold_complex() -> tensor<1x64x224x224xcomplex<f32>> {
  %cst = mhlo.constant dense<(0.000000e+00,1.000000e+00)> : tensor<complex<f32>>
  %b = "mhlo.broadcast_in_dim"(%cst) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<complex<f32>>) -> tensor<1x64x224x224xcomplex<f32>>
  func.return %b : tensor<1x64x224x224xcomplex<f32>>
}
// CHECK-NEXT: %[[CST:.*]] = mhlo.constant dense<(0.000000e+00,1.000000e+00)> : tensor<1x64x224x224xcomplex<f32>>
// CHECK-NEXT: return %[[CST]] : tensor<1x64x224x224xcomplex<f32>>


// CHECK-LABEL: @complex_expand_fold
func.func @complex_expand_fold(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = "mhlo.complex"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xcomplex<f32>>)
  %1 = mhlo.real(%0) : (tensor<4xcomplex<f32>>) -> (tensor<4xf32>)
  %2 = "mhlo.imag"(%0) : (tensor<4xcomplex<f32>>) -> (tensor<4xf32>)
  // CHECK: return %arg0, %arg1
  func.return %1, %2 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: @complex_collapse_fold
func.func @complex_collapse_fold(%arg0: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %0 = mhlo.real(%arg0) : (tensor<4xcomplex<f32>>) -> (tensor<4xf32>)
  %1 = "mhlo.imag"(%arg0) : (tensor<4xcomplex<f32>>) -> (tensor<4xf32>)
  %2 = "mhlo.complex"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xcomplex<f32>>
  // CHECK: return %arg0
  func.return %2 : tensor<4xcomplex<f32>>
}

// CHECK-LABEL: @dynamic_iota_is_static
func.func @dynamic_iota_is_static(%arg0 : tensor<1xindex>) -> tensor<4xi32> {
  // CHECK: [[RESULT:%.*]] = "mhlo.iota"
  // CHECK: return [[RESULT]]
  %0 = "mhlo.dynamic_iota"(%arg0) {iota_dimension = 0 : i64} : (tensor<1xindex>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: @dynamic_iota_broadcast
func.func @dynamic_iota_broadcast(%arg0 : tensor<2xindex>) -> tensor<5x?xi32> {
  // CHECK: [[IOTA:%.+]] = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<5xi32>
  // CHECK: [[BROADCAST:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[IOTA]], %arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<5xi32>, tensor<2xindex>) -> tensor<5x?xi32>
  %0 = "mhlo.dynamic_iota"(%arg0) {iota_dimension = 0 : i64} : (tensor<2xindex>) -> tensor<5x?xi32>

  // CHECK: return [[BROADCAST]]
  func.return %0 : tensor<5x?xi32>
}

// CHECK-LABEL: @dynamic_iota_broadcast_second
func.func @dynamic_iota_broadcast_second(%arg0 : tensor<2xindex>) -> tensor<5x?xi32> {
  // CHECK-NEXT: [[CAST1:%.+]] = arith.index_cast %arg0 : tensor<2xindex> to tensor<2xi64>
  // CHECK-NEXT: [[SLICE:%.+]] = "mhlo.slice"([[CAST1]]) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK-NEXT: [[CAST2:%.+]] = arith.index_cast [[SLICE]] : tensor<1xi64> to tensor<1xindex>
  // CHECK-NEXT: [[IOTA:%.+]] = "mhlo.dynamic_iota"([[CAST2]]) {iota_dimension = 0 : i64} : (tensor<1xindex>) -> tensor<?xi32>
  // CHECK-NEXT: [[BROADCAST:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[IOTA]], %arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xi32>, tensor<2xindex>) -> tensor<5x?xi32>
  %0 = "mhlo.dynamic_iota"(%arg0) {iota_dimension = 1 : i64} : (tensor<2xindex>) -> tensor<5x?xi32>

  // CHECK: return [[BROADCAST]]
  func.return %0 : tensor<5x?xi32>
}

// CHECK-LABEL: @dynamic_iota_constant
func.func @dynamic_iota_constant(%arg0 : tensor<2xindex>) -> tensor<1x?xi32> {
  // CHECK: [[IOTA:%.+]] = mhlo.constant dense<0> : tensor<1xi32>
  // CHECK: [[BROADCAST:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[IOTA]], %arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi32>, tensor<2xindex>) -> tensor<1x?xi32>
  %0 = "mhlo.dynamic_iota"(%arg0) {iota_dimension = 0 : i64} : (tensor<2xindex>) -> tensor<1x?xi32>

  // CHECK: return [[BROADCAST]]
  func.return %0 : tensor<1x?xi32>
}

// CHECK-LABEL: @iota_constant
func.func @iota_constant() -> tensor<1xi32> {
  // CHECK: [[CONST:%.+]] = mhlo.constant dense<0> : tensor<1xi32>
  %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<1xi32>

  // CHECK: return [[CONST]] : tensor<1xi32>
  func.return %0 : tensor<1xi32>
}

// CHECK-LABEL: @iota_constant_multi
func.func @iota_constant_multi() -> tensor<1x4xi32> {
  // CHECK: [[CONST:%.+]] = mhlo.constant dense<0> : tensor<1x4xi32>
  %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<1x4xi32>

  // CHECK: return [[CONST]] : tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// CHECK-LABEL: @iota_not_lowered_to_constant
func.func @iota_not_lowered_to_constant() -> tensor<4xi32> {
  // CHECK: [[RESULT:%.*]] = "mhlo.iota"
  // CHECK: return [[RESULT]]
  %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: @iota_broadcast
func.func @iota_broadcast() -> tensor<5x4xi32> {
  // CHECK: [[IOTA:%.+]] = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<5xi32>
  // CHECK: [[RESULT:%.+]] = "mhlo.broadcast_in_dim"([[IOTA]]) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<5xi32>) -> tensor<5x4xi32>
  %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<5x4xi32>

  func.return %0 : tensor<5x4xi32>
}

// CHECK-LABEL: @iota_broadcast
func.func @iota_broadcast_second() -> tensor<5x4xi32> {
  // CHECK: [[IOTA:%.+]] = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4xi32>
  // CHECK: [[RESULT:%.+]] = "mhlo.broadcast_in_dim"([[IOTA]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xi32>) -> tensor<5x4xi32>
  %0 = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<5x4xi32>

  func.return %0 : tensor<5x4xi32>
}

// CHECK-LABEL: @unary_einsum
func.func @unary_einsum(%arg0: tensor<2x3xf32>) -> tensor<2x2xf32> {
  // CHECK: %[[ONE:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: "mhlo.einsum"(%[[ONE]], %arg0) {einsum_config = ",ab->aa"}
  %0 = "mhlo.unary_einsum"(%arg0) {einsum_config = "ab->aa"} : (tensor<2x3xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func @fold_copy
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @fold_copy(%arg : tensor<1x4xf32>) -> tensor<1x4xf32> {
  // CHECK: return [[ARG]]
  %0 = "mhlo.copy"(%arg) : (tensor<1x4xf32>) -> tensor<1x4xf32>
  func.return %0 : tensor<1x4xf32>
}

// CHECK-LABEL: func @dynamic_reshape_not_actually_dynamic
func.func @dynamic_reshape_not_actually_dynamic(%arg0: tensor<4xf32>, %shape: tensor<2xindex>) -> tensor<4x1xf32> {
  // CHECK: mhlo.reshape
  %0 = "mhlo.dynamic_reshape"(%arg0, %shape) : (tensor<4xf32>, tensor<2xindex>) -> tensor<4x1xf32>
  func.return %0 : tensor<4x1xf32>
}

// CHECK-LABEL: func @shape_of_dynamic_reshape
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
func.func @shape_of_dynamic_reshape(%arg0: tensor<*xf32>, %shape: tensor<2xindex>) -> tensor<2xindex> {
  // CHECK: return [[ARG1]]
  %0 = "mhlo.dynamic_reshape"(%arg0, %shape) : (tensor<*xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %1 = shape.shape_of %0 : tensor<?x?xf32> -> tensor<2xindex>
  func.return %1 : tensor<2xindex>
}

// CHECK-LABEL: func @dynamic_reshape_rank_1_to_rank_1
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
func.func @dynamic_reshape_rank_1_to_rank_1(%arg0: tensor<?xcomplex<f32>>,
    %shape: tensor<?xindex>) -> tensor<?xf32> {
  // CHECK: [[RES:%[a-zA-Z0-9]+]] = mhlo.real([[ARG0]]) : (tensor<?xcomplex<f32>>) -> tensor<?xf32>
  // CHECK: return [[RES]]
  %0 = mhlo.real(%arg0): (tensor<?xcomplex<f32>>) -> tensor<?xf32>
  %1 = shape.shape_of %arg0 : tensor<?xcomplex<f32>> -> tensor<1xindex>
  %2 = shape.num_elements %1 : tensor<1xindex> -> index
  %3 = tensor.from_elements %2 : tensor<1xindex>
  %4 = "mhlo.dynamic_reshape"(%0, %3)
    : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
  func.return %4 : tensor<?xf32>
}

// CHECK-LABEL: func @dynamic_reshape_of_dynamic_reshape
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
func.func @dynamic_reshape_of_dynamic_reshape(%arg0: tensor<?xf16>, %shape: tensor<?xindex>) -> tensor<?xf16> {
  // CHECK: return [[ARG0]]
  %0 = "mhlo.dynamic_reshape"(%arg0, %shape) : (tensor<?xf16>, tensor<?xindex>) -> tensor<*xf16>
  %1 = shape.shape_of %0 : tensor<*xf16> -> tensor<?xindex>
  %2 = shape.num_elements %1 : tensor<?xindex> -> index
  %3 = tensor.from_elements %2 : tensor<1xindex>
  %4 = "mhlo.dynamic_reshape"(%0, %3) : (tensor<*xf16>, tensor<1xindex>) -> tensor<?xf16>
  func.return %4 : tensor<?xf16>
}

// CHECK-LABEL: do_not_dce_while_with_outfeed
func.func @do_not_dce_while_with_outfeed(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK: mhlo.while
  %0 = "mhlo.while"(%arg0) ({
  ^bb0(%arg1: tensor<i64>):
    %1 = "mhlo.compare"(%arg1, %arg1) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i64>):
    %1 = "mhlo.create_token"() : () -> !mhlo.token
    // Side-effecting op outfeed present inside while.
    %2 = "mhlo.outfeed"(%arg1, %1) {outfeed_config = ""} : (tensor<i64>, !mhlo.token) -> !mhlo.token
    "mhlo.return"(%arg1) : (tensor<i64>) -> ()
  }) : (tensor<i64>) -> tensor<i64>

  func.return %arg0 : tensor<i64>
}

// CHECK-LABEL: dce_while_without_side_effect
func.func @dce_while_without_side_effect(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK-NOT: mhlo.while
  %0 = "mhlo.while"(%arg0) ({
  ^bb0(%arg1: tensor<i64>):
    %1 = "mhlo.compare"(%arg1, %arg1) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i64>):
    %1 = "mhlo.create_token"() : () -> !mhlo.token
    "mhlo.return"(%arg1) : (tensor<i64>) -> ()
  }) : (tensor<i64>) -> tensor<i64>

  func.return %arg0 : tensor<i64>
}

// CHECK-LABEL: fold_sign_posi
func.func @fold_sign_posi() -> tensor<i32> {
  // CHECK: %0 = mhlo.constant dense<1> : tensor<i32>
  %0 = mhlo.constant dense<2> : tensor<i32>
  %1 = "mhlo.sign"(%0) : (tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK-LABEL: fold_sign_negi
func.func @fold_sign_negi() -> tensor<i32> {
  // CHECK: %0 = mhlo.constant dense<-1> : tensor<i32>
  %0 = mhlo.constant dense<-2> : tensor<i32>
  %1 = "mhlo.sign"(%0) : (tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK-LABEL: fold_sign_posf
func.func @fold_sign_posf() -> tensor<bf16> {
  // CHECK: %0 = mhlo.constant dense<1.000000e+00> : tensor<bf16>
  %0 = mhlo.constant dense<2.000000e+00> : tensor<bf16>
  %1 = "mhlo.sign"(%0) : (tensor<bf16>) -> tensor<bf16>
  func.return %1 : tensor<bf16>
}

// CHECK-LABEL: fold_sign_negf
func.func @fold_sign_negf() -> tensor<bf16> {
  // CHECK: %0 = mhlo.constant dense<-1.000000e+00> : tensor<bf16>
  %0 = mhlo.constant dense<-2.000000e+00> : tensor<bf16>
  %1 = "mhlo.sign"(%0) : (tensor<bf16>) -> tensor<bf16>
  func.return %1 : tensor<bf16>
}

// CHECK-LABEL: fold_sign_negzf
func.func @fold_sign_negzf() -> tensor<bf16> {
  // CHECK: %0 = mhlo.constant dense<-0.000000e+00> : tensor<bf16>
  %0 = mhlo.constant dense<-0.000000e+00> : tensor<bf16>
  %1 = "mhlo.sign"(%0) : (tensor<bf16>) -> tensor<bf16>
  func.return %1 : tensor<bf16>
}

// CHECK-LABEL: fold_compare_same_eq
func.func @fold_compare_same_eq(%arg0: tensor<i64>) -> tensor<i1> {
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %0 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = #mhlo<"comparison_direction EQ">} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// CHECK-LABEL: fold_compare_same_le
func.func @fold_compare_same_le(%arg0: tensor<i64>) -> tensor<i1> {
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %0 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = #mhlo<"comparison_direction LE">} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// CHECK-LABEL: fold_compare_same_ge
func.func @fold_compare_same_ge(%arg0: tensor<i64>) -> tensor<i1> {
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %0 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = #mhlo<"comparison_direction GE">} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: fold_compare_same_ne
func.func @fold_compare_same_ne(%arg0: tensor<i64>) -> tensor<i1> {
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %0 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = #mhlo<"comparison_direction NE">} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// CHECK-LABEL: fold_compare_same_lt
func.func @fold_compare_same_lt(%arg0: tensor<i64>) -> tensor<i1> {
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %0 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// CHECK-LABEL: fold_compare_same_gt
func.func @fold_compare_same_gt(%arg0: tensor<i64>) -> tensor<i1> {
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %0 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = #mhlo<"comparison_direction GT">} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// Address NaN != NaN.
// CHECK-LABEL: dont_fold_compare_same_eq_float
func.func @dont_fold_compare_same_eq_float(%arg0: tensor<f16>) -> tensor<i1> {
  // CHECK: %0 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = #mhlo<"comparison_direction EQ">} : (tensor<f16>, tensor<f16>) -> tensor<i1>
  %0 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = #mhlo<"comparison_direction EQ">} : (tensor<f16>, tensor<f16>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// Address NaN != NaN for complex types.
// CHECK-LABEL: dont_fold_compare_same_eq_complex
func.func @dont_fold_compare_same_eq_complex(%arg0: tensor<complex<f32>>) -> tensor<i1> {
  // CHECK: %0 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = #mhlo<"comparison_direction EQ">} : (tensor<complex<f32>>, tensor<complex<f32>>) -> tensor<i1>
  %0 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = #mhlo<"comparison_direction EQ">} : (tensor<complex<f32>>, tensor<complex<f32>>) -> tensor<i1>
  func.return %0 : tensor<i1>
}


// CHECK-LABEL: fold_compare_false_eq
func.func @fold_compare_false_eq() -> tensor<i1> {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction EQ">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}
// CHECK-LABEL: fold_compare_true_eq
func.func @fold_compare_true_eq() -> tensor<i1> {
  %0 = mhlo.constant dense<1> : tensor<i32>
  %1 = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction EQ">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_bools_true_eq
func.func @fold_compare_bools_true_eq(%arg : tensor<i1>) -> tensor<i1> {
  %1 = mhlo.constant dense<true> : tensor<i1>
  // CHECK: return %arg
  %2 = "mhlo.compare"(%arg, %1) {comparison_direction = #mhlo<"comparison_direction EQ">} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_eq_float
func.func @fold_compare_false_eq_float() -> tensor<i1> {
  %0 = mhlo.constant dense<0.> : tensor<f32>
  %1 = mhlo.constant dense<1.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction EQ">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_eq_float
func.func @fold_compare_true_eq_float() -> tensor<i1> {
  %0 = mhlo.constant dense<1.> : tensor<f32>
  %1 = mhlo.constant dense<1.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction EQ">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_ne
func.func @fold_compare_false_ne() -> tensor<i1> {
  %0 = mhlo.constant dense<1> : tensor<i32>
  %1 = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction NE">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_ne
func.func @fold_compare_true_ne() -> tensor<i1> {
  %0 = mhlo.constant dense<1> : tensor<i32>
  %1 = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction NE">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_bools_false_ne
func.func @fold_compare_bools_false_ne(%arg : tensor<i1>) -> tensor<i1> {
  %1 = mhlo.constant dense<false> : tensor<i1>
  // CHECK: return %arg
  %2 = "mhlo.compare"(%arg, %1) {comparison_direction = #mhlo<"comparison_direction NE">} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_ne_float
func.func @fold_compare_false_ne_float() -> tensor<i1> {
  %0 = mhlo.constant dense<1.> : tensor<f32>
  %1 = mhlo.constant dense<1.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction NE">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_ne_float
func.func @fold_compare_true_ne_float() -> tensor<i1> {
  %0 = mhlo.constant dense<0.> : tensor<f32>
  %1 = mhlo.constant dense<1.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction NE">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_lt
func.func @fold_compare_false_lt() -> tensor<i1> {
  %0 = mhlo.constant dense<1> : tensor<i32>
  %1 = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_lt
func.func @fold_compare_true_lt() -> tensor<i1> {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_lt_float
func.func @fold_compare_false_lt_float() -> tensor<i1> {
  %0 = mhlo.constant dense<1.> : tensor<f32>
  %1 = mhlo.constant dense<1.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_lt_float
func.func @fold_compare_true_lt_float() -> tensor<i1> {
  %0 = mhlo.constant dense<0.> : tensor<f32>
  %1 = mhlo.constant dense<1.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_le
func.func @fold_compare_false_le() -> tensor<i1> {
  %0 = mhlo.constant dense<1> : tensor<i32>
  %1 = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction LE">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_le
func.func @fold_compare_true_le() -> tensor<i1> {
  %0 = mhlo.constant dense<1> : tensor<i32>
  %1 = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction LE">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_le_float
func.func @fold_compare_false_le_float() -> tensor<i1> {
  %0 = mhlo.constant dense<1.> : tensor<f32>
  %1 = mhlo.constant dense<0.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction LE">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_le_float
func.func @fold_compare_true_le_float() -> tensor<i1> {
  %0 = mhlo.constant dense<1.> : tensor<f32>
  %1 = mhlo.constant dense<1.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction LE">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_gt
func.func @fold_compare_false_gt() -> tensor<i1> {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction GT">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_gt
func.func @fold_compare_true_gt() -> tensor<i1> {
  %0 = mhlo.constant dense<1> : tensor<i32>
  %1 = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction GT">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_gt_float
func.func @fold_compare_false_gt_float() -> tensor<i1> {
  %0 = mhlo.constant dense<0.> : tensor<f32>
  %1 = mhlo.constant dense<0.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction GT">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_gt_float
func.func @fold_compare_true_gt_float() -> tensor<i1> {
  %0 = mhlo.constant dense<1.> : tensor<f32>
  %1 = mhlo.constant dense<0.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction GT">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_ge
func.func @fold_compare_false_ge() -> tensor<i1> {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant dense<1> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction GE">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_ge
func.func @fold_compare_true_ge() -> tensor<i1> {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant dense<0> : tensor<i32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction GE">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_false_ge_float
func.func @fold_compare_false_ge_float() -> tensor<i1> {
  %0 = mhlo.constant dense<0.> : tensor<f32>
  %1 = mhlo.constant dense<1.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction GE">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: fold_compare_true_ge_float
func.func @fold_compare_true_ge_float() -> tensor<i1> {
  %0 = mhlo.constant dense<0.> : tensor<f32>
  %1 = mhlo.constant dense<0.> : tensor<f32>
  // CHECK: %0 = mhlo.constant dense<true> : tensor<i1>
  %2 = "mhlo.compare"(%0, %1) {comparison_direction = #mhlo<"comparison_direction GE">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

// CHECK-LABEL: unpack_repack_same_tuple
// CHECK-SAME: ([[ARG0:%.*]]: tuple<tensor<i32>, !mhlo.token, tensor<f32>>)
func.func @unpack_repack_same_tuple(%arg0: tuple<tensor<i32>, !mhlo.token, tensor<f32>>) -> tuple<tensor<i32>, !mhlo.token, tensor<f32>> {
  %0 = "mhlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<i32>, !mhlo.token, tensor<f32>>) -> tensor<i32>
  %1 = "mhlo.get_tuple_element"(%arg0) {index = 1 : i32} : (tuple<tensor<i32>, !mhlo.token, tensor<f32>>) -> !mhlo.token
  %2 = "mhlo.get_tuple_element"(%arg0) {index = 2 : i32} : (tuple<tensor<i32>, !mhlo.token, tensor<f32>>) -> tensor<f32>
  %3 = "mhlo.tuple"(%0, %1, %2) : (tensor<i32>, !mhlo.token, tensor<f32>) -> tuple<tensor<i32>, !mhlo.token, tensor<f32>>

  // CHECK: return [[ARG0]]
  func.return %3 : tuple<tensor<i32>, !mhlo.token, tensor<f32>>
}

// CHECK-LABEL: unpack_repack_same_tuple_single_element
// CHECK-SAME: ([[ARG0:%.*]]: tuple<tensor<i32>>)
func.func @unpack_repack_same_tuple_single_element(%arg0: tuple<tensor<i32>>) -> tuple<tensor<i32>> {
  %0 = "mhlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<i32>>) -> tensor<i32>
  %3 = "mhlo.tuple"(%0) : (tensor<i32>) -> tuple<tensor<i32>>

  // CHECK: return [[ARG0]]
  func.return %3 : tuple<tensor<i32>>
}

// CHECK-LABEL: func @erase_dead_lhlo_constant
func.func @erase_dead_lhlo_constant() {
  %M = memref.alloc() : memref<256x1024xf32>
  // CHECK-NEXT: return
  "lmhlo.constant"(%M) {value = dense<0.0> : tensor<f32>} : (memref<256x1024xf32>) -> ()
  memref.dealloc %M : memref<256x1024xf32>
  func.return
}

// A negative test for dead lhlo constant op erasure.
// CHECK-LABEL: func @erase_dead_lhlo_constant_negative
func.func @erase_dead_lhlo_constant_negative(%M : memref<4xf32>) -> memref<256x1024xf32> {
  // CHECK-NEXT: lmhlo.constant
  "lmhlo.constant"(%M) {value = dense<0.0> : tensor<f32>} : (memref<4xf32>) -> ()
  // CHECK-NEXT: memref.alloc
  // CHECK-NEXT: lmhlo.constant
  %N = memref.alloc() : memref<256x1024xf32>
  "lmhlo.constant"(%N) {value = dense<0.0> : tensor<f32>} : (memref<256x1024xf32>) -> ()
  func.return %N : memref<256x1024xf32>
}

// CHECK-LABEL: func @fold_get_dimension_size
func.func @fold_get_dimension_size(%I: tensor<1x128x512xf32>) -> tensor<i32> {
  %size = "mhlo.get_dimension_size"(%I) {dimension = 2 : i64} : (tensor<1x128x512xf32>) -> tensor<i32>
  func.return %size : tensor<i32>
  // CHECK-NEXT: %[[C:.*]] = mhlo.constant dense<512> : tensor<i32>
  // CHECK-NEXT: return %[[C]]
}

// CHECK-LABEL: func @fold_get_dimension_size_fail
func.func @fold_get_dimension_size_fail(%I: tensor<1x128x?xf32>) -> tensor<i32> {
  // CHECK: "mhlo.get_dimension_size"
  %size = "mhlo.get_dimension_size"(%I) {dimension = 2 : i64} : (tensor<1x128x?xf32>) -> tensor<i32>
  func.return %size : tensor<i32>
}

// CHECK-LABEL: func @fold_set_dimension_size
// CHECK-SAME: (%[[I:.*]]: tensor<1x128x512xf32>)
func.func @fold_set_dimension_size(%I: tensor<1x128x512xf32>) -> tensor<1x128x512xf32> {
  %dim = mhlo.constant dense<512> : tensor<i32>
  %result = "mhlo.set_dimension_size"(%I, %dim) {dimension = 2 : i64} : (tensor<1x128x512xf32>, tensor<i32>) -> tensor<1x128x512xf32>
  func.return %result : tensor<1x128x512xf32>

  // CHECK-NEXT: return %[[I]]
}

// CHECK-LABEL: func @fold_select_same
func.func @fold_select_same(%arg0 : tensor<f32>, %arg1 : tensor<i1>) -> tensor<f32> {
  %1 = "mhlo.select"(%arg1, %arg0, %arg0) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: return %arg0
  func.return %1 : tensor<f32>
}

// CHECK-LABEL: func @fold_select_first
func.func @fold_select_first(%arg0 : tensor<f32>, %arg1 : tensor<f32>) -> tensor<f32> {
  %0 = mhlo.constant dense<1> : tensor<i1>
  %1 = "mhlo.select"(%0, %arg0, %arg1) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: return %arg0
  func.return %1 : tensor<f32>
}

// CHECK-LABEL: func @fold_select_second
func.func @fold_select_second(%arg0 : tensor<f32>, %arg1 : tensor<f32>) -> tensor<f32> {
  %0 = mhlo.constant dense<0> : tensor<i1>
  %1 = "mhlo.select"(%0, %arg0, %arg1) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: return %arg1
  func.return %1 : tensor<f32>
}

// CHECK-LABEL: func @fold_select_vector
func.func @fold_select_vector(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = mhlo.constant dense<1> : tensor<4xi1>
  %1 = "mhlo.select"(%0, %arg0, %arg1) : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK: return %arg0
  func.return %1 : tensor<4xf32>
}

// CHECK-LABEL: func @simplify_not_as_select_pred(
// CHECK-SAME: [[ARGV0:%[a-zA-Z0-9_]+]]: tensor<4xi1>
// CHECK-SAME: [[ARGV1:%[a-zA-Z0-9_]+]]: tensor<4xf32>
// CHECK-SAME: [[ARGV2:%[a-zA-Z0-9_]+]]: tensor<4xf32>
func.func @simplify_not_as_select_pred(%arg0 : tensor<4xi1>, %arg1 : tensor<4xf32>, %arg2 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = "mhlo.not"(%arg0) : (tensor<4xi1>) -> tensor<4xi1>
  %1 = "mhlo.select"(%0, %arg1, %arg2) : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK: "mhlo.select"([[ARGV0]], [[ARGV2]], [[ARGV1]])
  func.return %1 : tensor<4xf32>
}

// CHECK-LABEL: gather_to_slice
func.func @gather_to_slice(%arg0: tensor<5x6x7xf32>) -> tensor<3x6x5xf32> {
  %0 = arith.constant dense<[1, 2]> : tensor<2xi32>
  %1 = "mhlo.gather"(%arg0, %0) {
    dimension_numbers = #mhlo.gather<
      index_vector_dim = 0,
      offset_dims = [0, 1, 2],
      start_index_map = [0, 2],
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[3, 6, 5]> : tensor<3xi64>} : (tensor<5x6x7xf32>, tensor<2xi32>) -> tensor<3x6x5xf32>
  func.return %1 : tensor<3x6x5xf32>
  // CHECK:  %[[RET:.*]] = "mhlo.slice"(%arg0) {limit_indices = dense<[4, 6, 7]> : tensor<3xi64>, start_indices = dense<[1, 0, 2]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<5x6x7xf32>) -> tensor<3x6x5xf32>
  // CHECK: return %[[RET]] : tensor<3x6x5xf32>
}

// CHECK-LABEL: gather_scalar_index_to_slice
func.func @gather_scalar_index_to_slice(%arg0: tensor<5x6x7xf32>) -> tensor<5x6x4xf32> {
  %0 = arith.constant dense<1> : tensor<i32>
  %1 = "mhlo.gather"(%arg0, %0) {
    dimension_numbers = #mhlo.gather<
      index_vector_dim = 0,
      offset_dims = [0, 1, 2],
      start_index_map = [2],
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[5, 6, 4]> : tensor<3xi64>} : (tensor<5x6x7xf32>, tensor<i32>) -> tensor<5x6x4xf32>
  func.return %1 : tensor<5x6x4xf32>
  // CHECK:  %[[RET:.*]] = "mhlo.slice"(%arg0) {limit_indices = dense<[5, 6, 5]> : tensor<3xi64>, start_indices = dense<[0, 0, 1]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<5x6x7xf32>) -> tensor<5x6x4xf32>
  // CHECK: return %[[RET]] : tensor<5x6x4xf32>
}

// CHECK-LABEL: gather_to_slice_reshape
func.func @gather_to_slice_reshape(%arg0: tensor<5x6x7xf32>) -> tensor<3x6xf32> {
  %0 = arith.constant dense<[1, 2]> : tensor<2xi32>
  %1 = "mhlo.gather"(%arg0, %0) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [2],
      index_vector_dim = 0,
      offset_dims = [0, 1],
      start_index_map = [0, 2],
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[3, 6, 1]> : tensor<3xi64>} : (tensor<5x6x7xf32>, tensor<2xi32>) -> tensor<3x6xf32>
  func.return %1 : tensor<3x6xf32>
  // CHECK:  %[[V0:.*]] = "mhlo.slice"(%arg0) {limit_indices = dense<[4, 6, 3]> : tensor<3xi64>, start_indices = dense<[1, 0, 2]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<5x6x7xf32>) -> tensor<3x6x1xf32>
  // CHECK:  %[[V1:.*]] = "mhlo.reshape"(%[[V0]]) : (tensor<3x6x1xf32>) -> tensor<3x6xf32>
  // CHECK: return %[[V1]] : tensor<3x6xf32>
}

// CHECK-LABEL: gather_to_slice_indices_clamp_upperbound
func.func @gather_to_slice_indices_clamp_upperbound(%arg0 : tensor<4x2xui32>) -> tensor<2xui32> {
  %0 = arith.constant dense<4> : tensor<1xi32>
  %1 = "mhlo.gather"(%arg0, %0) {
    dimension_numbers = #mhlo.gather<
      offset_dims = [0],
      index_vector_dim = 0,
      collapsed_slice_dims = [0],
      start_index_map = [0]
    >, indices_are_sorted = true,
    slice_sizes = dense<[1, 2]> : tensor<2xi64>} : (tensor<4x2xui32>, tensor<1xi32>) -> tensor<2xui32>
  func.return %1 : tensor<2xui32>
  // CHECK:  %[[V0:.*]] = "mhlo.slice"(%arg0) {limit_indices = dense<[4, 2]> : tensor<2xi64>, start_indices = dense<[3, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x2xui32>) -> tensor<1x2xui32>
  // CHECK:  %[[V1:.*]] = "mhlo.reshape"(%[[V0]]) : (tensor<1x2xui32>) -> tensor<2xui32>
  // CHECK: return %[[V1]] : tensor<2xui32>
}

// CHECK-LABEL: gather_to_slice_indices_clamp_lowerbound
func.func @gather_to_slice_indices_clamp_lowerbound(%arg0 : tensor<4x2xui32>) -> tensor<2xui32> {
  %0 = arith.constant dense<-1> : tensor<1xi32>
  %1 = "mhlo.gather"(%arg0, %0) {
    dimension_numbers = #mhlo.gather<
      offset_dims = [0],
      index_vector_dim = 0,
      collapsed_slice_dims = [0],
      start_index_map = [0]
    >, indices_are_sorted = true,
    slice_sizes = dense<[1, 2]> : tensor<2xi64>} : (tensor<4x2xui32>, tensor<1xi32>) -> tensor<2xui32>
  func.return %1 : tensor<2xui32>
  // CHECK:  %[[V0:.*]] = "mhlo.slice"(%arg0) {limit_indices = dense<[1, 2]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x2xui32>) -> tensor<1x2xui32>
  // CHECK:  %[[V1:.*]] = "mhlo.reshape"(%[[V0]]) : (tensor<1x2xui32>) -> tensor<2xui32>
  // CHECK: return %[[V1]] : tensor<2xui32>
}

// CHECK-LABEL: func @fold_and_same
func.func @fold_and_same(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = "mhlo.and"(%arg0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: return %arg0
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_and_ones
func.func @fold_and_ones(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<-1> : tensor<4xi32>
  %1 = "mhlo.and"(%0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: return %arg0
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_and_zeros
func.func @fold_and_zeros(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<0> : tensor<4xi32>
  %1 = "mhlo.and"(%0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: return %0
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_and_constant
func.func @fold_and_constant(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<7> : tensor<4xi32>
  // CHECK: mhlo.and
  %1 = "mhlo.and"(%0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_and_constants
func.func @fold_and_constants() -> tensor<4xi32> {
  %0 = mhlo.constant dense<[0, 1, 6, 3]> : tensor<4xi32>
  %1 = mhlo.constant dense<[7, 3, 7, 2]> : tensor<4xi32>
  %2 = "mhlo.and"(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: %0 = mhlo.constant dense<[0, 1, 6, 2]> : tensor<4xi32>
  // CHECK: return %0
  func.return %2 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_or_same
func.func @fold_or_same(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = "mhlo.or"(%arg0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: return %arg0
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_or_ones
func.func @fold_or_ones(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<-1> : tensor<4xi32>
  %1 = "mhlo.or"(%0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: return %0
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_or_zeros
func.func @fold_or_zeros(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<0> : tensor<4xi32>
  %1 = "mhlo.or"(%0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: return %arg0
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_or_constant
func.func @fold_or_constant(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<7> : tensor<4xi32>
  // CHECK: mhlo.or
  %1 = "mhlo.or"(%0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_or_zeros_right
func.func @fold_or_zeros_right(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<0> : tensor<4xi32>
  %1 = "mhlo.or"(%arg0, %0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: return %arg0
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_or_zeros_constants
func.func @fold_or_zeros_constants() -> tensor<4xi32> {
  %0 = mhlo.constant dense<[0, 1, 6, 3]> : tensor<4xi32>
  %1 = mhlo.constant dense<[7, 3, 7, 2]> : tensor<4xi32>
  %2 = "mhlo.or"(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: %0 = mhlo.constant dense<[7, 3, 7, 3]> : tensor<4xi32>
  // CHECK: return %0
  func.return %2 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_xor_same
func.func @fold_xor_same(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = "mhlo.xor"(%arg0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: %0 = mhlo.constant dense<0> : tensor<4xi32>
  // CHECK: return %0
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_xor_same_dynamic
func.func @fold_xor_same_dynamic(%arg0 : tensor<?xi32>) -> tensor<?xi32> {
  %0 = "mhlo.xor"(%arg0, %arg0) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  // CHECK: mhlo.xor
  func.return %0 : tensor<?xi32>
}

// CHECK-LABEL: func @fold_xor_ones_left
func.func @fold_xor_ones_left(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<-1> : tensor<4xi32>
  // CHECK: mhlo.xor
  %1 = "mhlo.xor"(%0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_xor_ones_right
func.func @fold_xor_ones_right(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<-1> : tensor<4xi32>
  // CHECK: mhlo.xor
  %1 = "mhlo.xor"(%arg0, %0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_xor_zeros_left
func.func @fold_xor_zeros_left(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<0> : tensor<4xi32>
  %1 = "mhlo.xor"(%0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: return %arg0
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_xor_zeros_right
func.func @fold_xor_zeros_right(%arg0 : tensor<4xi32>) -> tensor<4xi32> {
  %0 = mhlo.constant dense<0> : tensor<4xi32>
  %1 = "mhlo.xor"(%arg0, %0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: return %arg0
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_xor_zeros_constants
func.func @fold_xor_zeros_constants() -> tensor<4xi32> {
  %0 = mhlo.constant dense<[0, 1, 6, 3]> : tensor<4xi32>
  %1 = mhlo.constant dense<[7, 3, 7, 2]> : tensor<4xi32>
  %2 = "mhlo.xor"(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  // CHECK: %0 = mhlo.constant dense<[7, 2, 1, 1]> : tensor<4xi32>
  // CHECK: return %0
  func.return %2 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_negate_int
func.func @fold_negate_int() -> tensor<4xi32> {
  %0 = mhlo.constant dense<[0, 1, 6, -3]> : tensor<4xi32>
  // CHECK: mhlo.constant dense<[0, -1, -6, 3]>
  %1 = "mhlo.negate"(%0) : (tensor<4xi32>) -> tensor<4xi32>
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL: func @fold_negate_float
func.func @fold_negate_float() -> tensor<4xf32> {
  %0 = mhlo.constant dense<[0., 1., 6., -3.]> : tensor<4xf32>
  // CHECK: mhlo.constant dense<[-0.000000e+00, -1.000000e+00, -6.000000e+00, 3.000000e+00]>
  %1 = "mhlo.negate"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

// CHECK-LABEL func @fold_not()
func.func @fold_not() -> tensor<2x2xi1> {
  %0 = mhlo.constant dense<[[true, false], [true, false]]> : tensor<2x2xi1>
  // CHECK{LITERAL}: mhlo.constant dense<[[false, true], [false, true]]> : tensor<2x2xi1>
  %1 = "mhlo.not"(%0) : (tensor<2x2xi1>) -> tensor<2x2xi1>
  func.return %1 : tensor<2x2xi1>
}

// CHECK-LABEL func @fold_not_i32()
func.func @fold_not_i32() -> tensor<2x2xi32> {
  %0 = mhlo.constant dense<[[42, -12], [1, 0]]> : tensor<2x2xi32>
  // CHECK-LITERAL: mhlo.constant dense<[[0, 0], [0, 1]]> : tensor<2x2xi32>
  %1 = "mhlo.not"(%0) : (tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %1 : tensor<2x2xi32>
}

// CHECK-LABEL: func @fold_sqrt_f32_constants
func.func @fold_sqrt_f32_constants() -> tensor<4xf32> {
  %0 = mhlo.constant dense<1.0> : tensor<4xf32>
  %1 = "mhlo.sqrt"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  //     CHECK: mhlo.constant dense<1.000000e+00> : tensor<4xf32>
  // CHECK-NOT: mhlo.sqrt
  func.return %1 : tensor<4xf32>
}

// CHECK-LABEL: func @fold_sqrt_f64_constants
func.func @fold_sqrt_f64_constants() -> tensor<4xf64> {
  %0 = mhlo.constant dense<[1.0, 4.0, 9.0, 16.0]> : tensor<4xf64>
  %1 = "mhlo.sqrt"(%0) : (tensor<4xf64>) -> tensor<4xf64>
  //     CHECK: mhlo.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf64>
  // CHECK-NOT: mhlo.sqrt
  func.return %1 : tensor<4xf64>
}

// CHECK-LABEL: func @not_fold_sqrt_neg_constants
func.func @not_fold_sqrt_neg_constants() -> tensor<4xf32> {
  %0 = mhlo.constant dense<-1.0> : tensor<4xf32>
  %1 = "mhlo.sqrt"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: mhlo.constant dense<-1.000000e+00> : tensor<4xf32>
  // CHECK: mhlo.sqrt
  func.return %1 : tensor<4xf32>
}

// CHECK-LABEL: func @fold_if_true(
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]
//  CHECK-SAME: )
func.func @fold_if_true(%arg0 : tensor<f32>, %arg1 : tensor<f32>) -> tensor<f32> {
  // CHECK-NOT: mhlo.if
  // CHECK: return %[[ARG0]]
  %true = mhlo.constant dense<true> : tensor<i1>
  %0 = "mhlo.if"(%true) ({
      "mhlo.return"(%arg0) : (tensor<f32>) -> ()
  },  {
      "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  }) : (tensor<i1>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: func @fold_if_false(
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]
//  CHECK-SAME: )
func.func @fold_if_false(%arg0 : tensor<f32>, %arg1 : tensor<f32>) -> tensor<f32> {
  // CHECK-NOT: mhlo.if
  // CHECK: return %[[ARG1]]
  %false = mhlo.constant dense<false> : tensor<i1>
  %0 = "mhlo.if"(%false) ({
      "mhlo.return"(%arg0) : (tensor<f32>) -> ()
  },  {
      "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  }) : (tensor<i1>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: func @fold_case(
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]
//  CHECK-SAME: )
func.func @fold_case(%arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>) -> tensor<f32> {
  // CHECK-NOT: mhlo.case
  // CHECK: return %[[ARG1]]
  %c1 = mhlo.constant dense<1> : tensor<i32>
  %0 = "mhlo.case"(%c1) ({
      "mhlo.return"(%arg0) : (tensor<f32>) -> ()
    },  {
      "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  },  {
      "mhlo.return"(%arg2) : (tensor<f32>) -> ()
  }) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: func @fold_case_negative_index(
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]
//  CHECK-SAME: )
func.func @fold_case_negative_index(%arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>) -> tensor<f32> {
  // CHECK-NOT: mhlo.case
  // CHECK: return %[[ARG2]]
  %m1000 = mhlo.constant dense<-1000> : tensor<i32>
  %0 = "mhlo.case"(%m1000) ({
      "mhlo.return"(%arg0) : (tensor<f32>) -> ()
    },  {
      "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  },  {
      "mhlo.return"(%arg2) : (tensor<f32>) -> ()
  }) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: func @fold_case_oob_index(
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]
//  CHECK-SAME: )
func.func @fold_case_oob_index(%arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>) -> tensor<f32> {
  // CHECK-NOT: mhlo.case
  // CHECK: return %[[ARG2]]
  %c1000 = mhlo.constant dense<1000> : tensor<i32>
  %0 = "mhlo.case"(%c1000) ({
      "mhlo.return"(%arg0) : (tensor<f32>) -> ()
    },  {
      "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  },  {
      "mhlo.return"(%arg2) : (tensor<f32>) -> ()
  }) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: @tensor_flow_scatter_v1_update
func.func @tensor_flow_scatter_v1_update() -> tensor<3x3xi32> {
  %0 = arith.constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi32>
  %1 = arith.constant dense<[0, 2]> : tensor<2xi32>
  %2 = arith.constant dense<[[10, 20, 30], [70, 80, 90]]> : tensor<2x3xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      "mhlo.return"(%arg1) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers = #mhlo.scatter<
          update_window_dims = [1],
          inserted_window_dims = [0],
          scatter_dims_to_operand_dims = [0],
          index_vector_dim = 1,
        >,
        unique_indices = false
    } : (tensor<3x3xi32>, tensor<2xi32>, tensor<2x3xi32>) -> tensor<3x3xi32>
  func.return %3 : tensor<3x3xi32>
  // CHECK: mhlo.constant dense<[
  // CHECK-SAME: [10, 20, 30], [4, 5, 6], [70, 80, 90]
  // CHECK-SAME: ]> : tensor<3x3xi32>
}

// CHECK-LABEL: @tensor_flow_scatter_v2_update
func.func @tensor_flow_scatter_v2_update() -> tensor<3x3xi32> {
  %0 = arith.constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi32>
  %1 = arith.constant dense<[0, 2]> : tensor<2xi32>
  %2 = arith.constant dense<[[10, 30], [40, 60], [70, 90]]> : tensor<3x2xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      "mhlo.return"(%arg1) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers = #mhlo.scatter<
          update_window_dims = [0],
          inserted_window_dims = [1],
          scatter_dims_to_operand_dims = [1],
          index_vector_dim = 1,
        >,
        unique_indices = false
    } : (tensor<3x3xi32>, tensor<2xi32>, tensor<3x2xi32>) -> tensor<3x3xi32>
  func.return %3 : tensor<3x3xi32>
  // CHECK: mhlo.constant dense<[
  // CHECK-SAME: [10, 2, 30], [40, 5, 60], [70, 8, 90]
  // CHECK-SAME: ]> : tensor<3x3xi32>
}

// CHECK-LABEL: @tensor_flow_scatter_add
func.func @tensor_flow_scatter_add() -> tensor<3x3xi32> {
  %0 = arith.constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi32>
  %1 = arith.constant dense<[0, 2]> : tensor<2xi32>
  %2 = arith.constant dense<[[10, 20, 30], [70, 80, 90]]> : tensor<2x3xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %4 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
      "mhlo.return"(%4) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers = #mhlo.scatter<
          update_window_dims = [1],
          inserted_window_dims = [0],
          scatter_dims_to_operand_dims = [0],
          index_vector_dim = 1,
        >,
        unique_indices = false
    } : (tensor<3x3xi32>, tensor<2xi32>, tensor<2x3xi32>) -> tensor<3x3xi32>
  func.return %3 : tensor<3x3xi32>
  // CHECK: mhlo.constant dense<[
  // CHECK-SAME: [11, 22, 33], [4, 5, 6], [77, 88, 99]
  // CHECK-SAME: ]> : tensor<3x3xi32>
}

// CHECK-LABEL: @tensor_flow_scatter_repeated
func.func @tensor_flow_scatter_repeated() -> tensor<3x3xi32> {
  %0 = arith.constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi32>
  %1 = arith.constant dense<[1, 1]> : tensor<2xi32>
  %2 = arith.constant dense<[[10, 20, 30], [70, 80, 90]]> : tensor<2x3xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %4 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
      "mhlo.return"(%4) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers = #mhlo.scatter<
          update_window_dims = [1],
          inserted_window_dims = [0],
          scatter_dims_to_operand_dims = [0],
          index_vector_dim = 1,
        >,
        unique_indices = false
    } : (tensor<3x3xi32>, tensor<2xi32>, tensor<2x3xi32>) -> tensor<3x3xi32>
  func.return %3 : tensor<3x3xi32>
  // CHECK: mhlo.constant dense<[
  // CHECK-SAME: [1, 2, 3], [84, 105, 126], [7, 8, 9]
  // CHECK-SAME: ]> : tensor<3x3xi32>
}

// CHECK-LABEL: @tensor_flow_scatter_multiple_batch
func.func @tensor_flow_scatter_multiple_batch() -> tensor<3x3xi32> {
  %0 = arith.constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi32>
  %1 = arith.constant dense<[[0, 2], [2, 1]]> : tensor<2x2xi32>
  %2 = arith.constant dense<[[[10, 30], [40, 60], [70, 90]], [[5, 5], [5, 5], [5, 5]]]> : tensor<2x3x2xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %4 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
      "mhlo.return"(%4) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers =  #mhlo.scatter<
          update_window_dims = [1],
          inserted_window_dims = [1],
          scatter_dims_to_operand_dims = [1],
          index_vector_dim = 2,
        >,
        unique_indices = false
    } : (tensor<3x3xi32>, tensor<2x2xi32>, tensor<2x3x2xi32>) -> tensor<3x3xi32>
  func.return %3 : tensor<3x3xi32>
  // CHECK: mhlo.constant dense<[
  // CHECK-SAME: [11, 7, 38], [44, 10, 71], [77, 13, 104]
  // CHECK-SAME: ]> : tensor<3x3xi32>
}

// CHECK-LABEL: @tensor_flow_scatter_nd
func.func @tensor_flow_scatter_nd() -> tensor<3x3x2xi32> {
  %0 = arith.constant dense<[[[-1, 1], [-2, 2], [-3, 3]], [[-4, 4], [-5, 5], [-6, 6]], [[-7, 7], [-8, 8], [-9, 9]]]> : tensor<3x3x2xi32>
  %1 = arith.constant dense<[[0, 0], [1, 0]]> : tensor<2x2xi32>
  %2 = arith.constant dense<[[-10, 10], [-40, 40]]> : tensor<2x2xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      "mhlo.return"(%arg1) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers =  #mhlo.scatter<
          update_window_dims = [1],
          inserted_window_dims = [0, 1],
          scatter_dims_to_operand_dims = [0, 1],
          index_vector_dim = 1,
        >,
        unique_indices = false
    } : (tensor<3x3x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<3x3x2xi32>
  func.return %3 : tensor<3x3x2xi32>
  // CHECK: mhlo.constant dense<[
  // CHECK-SAME: [-10, 10], [-2, 2], [-3, 3]
  // CHECK-SAME: [-40, 40], [-5, 5], [-6, 6]
  // CHECK-SAME: [-7, 7], [-8, 8], [-9, 9]
  // CHECK-SAME: ]> : tensor<3x3x2xi32>
}

// CHECK-LABEL: @tensor_flow_scatter_nd_index_vector
func.func @tensor_flow_scatter_nd_index_vector() -> tensor<3x3x2xi32> {
  %0 = arith.constant dense<[[[-1, 1], [-2, 2], [-3, 3]], [[-4, 4], [-5, 5], [-6, 6]], [[-7, 7], [-8, 8], [-9, 9]]]> : tensor<3x3x2xi32>
  %1 = arith.constant dense<[[0, 0], [1, 0]]> : tensor<2x2xi32>
  %2 = arith.constant dense<[[-10, 10], [-20, 20]]> : tensor<2x2xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      "mhlo.return"(%arg1) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers = #mhlo.scatter<
          update_window_dims = [1],
          inserted_window_dims = [0, 1],
          scatter_dims_to_operand_dims = [0, 1],
          index_vector_dim = 0,
        >,
        unique_indices = false
    } : (tensor<3x3x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<3x3x2xi32>
  func.return %3 : tensor<3x3x2xi32>
  // CHECK: mhlo.constant dense<[
  // CHECK-SAME: [-20, 20], [-10, 10], [-3, 3]
  // CHECK-SAME: [-4, 4], [-5, 5], [-6, 6]
  // CHECK-SAME: [-7, 7], [-8, 8], [-9, 9]
  // CHECK-SAME: ]> : tensor<3x3x2xi32>
}

// CHECK-LABEL: @scatter_batch_dus
func.func @scatter_batch_dus() -> tensor<3x3xi32> {
  %0 = arith.constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi32>
  %1 = arith.constant dense<[[2, 1], [1, 1]]> : tensor<2x2xi32>
  %2 = arith.constant dense<[[[10]], [[20]]]> : tensor<2x1x1xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      "mhlo.return"(%arg1) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers = #mhlo.scatter<
          update_window_dims = [1, 2],
          scatter_dims_to_operand_dims = [0, 1],
          index_vector_dim = 0,
        >,
        unique_indices = false
    } : (tensor<3x3xi32>, tensor<2x2xi32>, tensor<2x1x1xi32>) -> tensor<3x3xi32>
  func.return %3 : tensor<3x3xi32>
  // CHECK: mhlo.constant dense<[
  // CHECK-SAME: [1, 2, 3], [4, 20, 6], [7, 10, 9]
  // CHECK-SAME: ]> : tensor<3x3xi32>
}

// CHECK-LABEL: @scatter_no_update_window_dim
func.func @scatter_no_update_window_dim() -> tensor<3xi32> {
  %0 = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
  %1 = arith.constant dense<[[[0], [1]], [[2], [1]]]> : tensor<2x2x1xi32>
  %2 = arith.constant dense<[[10, 20], [30, 40]]> : tensor<2x2xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %4 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
      "mhlo.return"(%4) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers = #mhlo.scatter<
          inserted_window_dims = [0],
          scatter_dims_to_operand_dims = [0],
          index_vector_dim = 2,
        >,
        unique_indices = false
    } : (tensor<3xi32>, tensor<2x2x1xi32>, tensor<2x2xi32>) -> tensor<3xi32>
  func.return %3 : tensor<3xi32>
  // CHECK: mhlo.constant dense<[10, 61, 32]> : tensor<3xi32>
}

// CHECK-LABEL: @scatter_negative_index
func.func @scatter_negative_index() -> tensor<3x3xi32> {
  %0 = arith.constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi32>
  %1 = arith.constant dense<[0, -1]> : tensor<2xi32>
  %2 = arith.constant dense<[[10, 20, 30], [70, 80, 90]]> : tensor<2x3xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      "mhlo.return"(%arg1) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers = #mhlo.scatter<
          update_window_dims = [1],
          inserted_window_dims = [0],
          scatter_dims_to_operand_dims = [0],
          index_vector_dim = 1,
        >,
        unique_indices = false
    } : (tensor<3x3xi32>, tensor<2xi32>, tensor<2x3xi32>) -> tensor<3x3xi32>
  func.return %3 : tensor<3x3xi32>
  // CHECK: constant dense<{{\[}}[1, 2, 3], [4, 5, 6], [7, 8, 9]{{\]}}> : tensor<3x3xi32>
  // CHECK: "mhlo.scatter"
}

// CHECK-LABEL: @scatter_out_of_bound
func.func @scatter_out_of_bound() -> tensor<3x3xi32> {
  %0 = arith.constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi32>
  %1 = arith.constant dense<[1, 5]> : tensor<2xi32>
  %2 = arith.constant dense<[[10, 20, 30], [70, 80, 90]]> : tensor<2x3xi32>
  %3 = "mhlo.scatter"(%0, %1, %2) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      "mhlo.return"(%arg1) : (tensor<i32>) -> ()
    }) {indices_are_sorted = false,
        scatter_dimension_numbers = #mhlo.scatter<
          update_window_dims = [1],
          inserted_window_dims = [0],
          scatter_dims_to_operand_dims = [0],
          index_vector_dim = 1,
        >,
        unique_indices = false
    } : (tensor<3x3xi32>, tensor<2xi32>, tensor<2x3xi32>) -> tensor<3x3xi32>
  func.return %3 : tensor<3x3xi32>
  // CHECK: constant dense<{{\[}}[1, 2, 3], [4, 5, 6], [7, 8, 9]{{\]}}> : tensor<3x3xi32>
  // CHECK: "mhlo.scatter"
}

// CHECK-LABEL: @scatter_complex
func.func public @scatter_complex() -> tensor<1xcomplex<f32>> {
  %0 = mhlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
  %1 = mhlo.constant dense<0> : tensor<1xi32>
  %2 = mhlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<1xcomplex<f32>>
  %3 = "mhlo.scatter"(%2, %1, %0) ({
  ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
    "mhlo.return"(%arg1) : (tensor<complex<f32>>) -> ()
  }) {indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true} : (tensor<1xcomplex<f32>>, tensor<1xi32>, tensor<complex<f32>>) -> tensor<1xcomplex<f32>>
  func.return %3 : tensor<1xcomplex<f32>>
}
// CHECK: "mhlo.scatter"


// CHECK-LABEL: @pad_identity_fold
func.func @pad_identity_fold(%arg0: tensor<5x7xf32>) -> tensor<5x7xf32> {
  %0 = arith.constant dense<0.0> : tensor<f32>
  %1 = "mhlo.pad"(%arg0, %0) {
    edge_padding_low = dense<0> : tensor<2xi64>,
    edge_padding_high = dense<0> : tensor<2xi64>,
    interior_padding = dense<0> : tensor<2xi64>
  } : (tensor<5x7xf32>, tensor<f32>) -> tensor<5x7xf32>
  func.return %1 : tensor<5x7xf32>
  // CHECK: return %arg0 : tensor<5x7xf32>
}

// CHECK-LABEL: @pad_fold
func.func @pad_fold() -> tensor<4x5xi32> {
  %0 = arith.constant dense<[[2, 3], [4, 5]]> : tensor<2x2xi32>
  %1 = arith.constant dense<1> : tensor<i32>
  %3 = "mhlo.pad"(%0, %1) {
    edge_padding_low = dense<[1, 0]> : tensor<2xi64>,
    edge_padding_high = dense<[1, 2]> : tensor<2xi64>,
    interior_padding = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<2x2xi32>, tensor<i32>) -> tensor<4x5xi32>
  func.return %3 : tensor<4x5xi32>
  // CHECK: constant dense<[
  // CHECK-SAME: [1, 1, 1, 1, 1], [2, 1, 3, 1, 1], [4, 1, 5, 1, 1], [1, 1, 1, 1, 1]
  // CHECK-SAME: ]> : tensor<4x5xi32>
}

// CHECK-LABEL: @pad_negative_fold
func.func @pad_negative_fold() -> tensor<4x4xi32> {
  %0 = arith.constant dense<[[2, 3], [4, 5]]> : tensor<2x2xi32>
  %1 = arith.constant dense<1> : tensor<i32>
  %3 = "mhlo.pad"(%0, %1) {
    edge_padding_low = dense<[1, -1]> : tensor<2xi64>,
    edge_padding_high = dense<[1, 2]> : tensor<2xi64>,
    interior_padding = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<2x2xi32>, tensor<i32>) -> tensor<4x4xi32>
  func.return %3 : tensor<4x4xi32>
  // CHECK: "mhlo.pad"
}

// CHECK-LABEL: @pad_fold_zero_elements
func.func @pad_fold_zero_elements() -> tensor<3xi32> {
  %0 = mhlo.constant dense<> : tensor<0xi32>
  %1 = mhlo.constant dense<7> : tensor<i32>
  %2 = "mhlo.pad"(%0, %1) {edge_padding_high = dense<3> : tensor<1xi64>, edge_padding_low = dense<0> : tensor<1xi64>, interior_padding = dense<0> : tensor<1xi64>} : (tensor<0xi32>, tensor<i32>) -> tensor<3xi32>
  func.return %2 : tensor<3xi32>
  // CHECK: mhlo.constant dense<7> : tensor<3xi32>
}

// CHECK-LABEL: @pad_float_fold
func.func @pad_float_fold() -> tensor<2xf32> {
  %0 = mhlo.constant dense<2.000000e+00> : tensor<1xf32>
  %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
  %2 = "mhlo.pad"(%0, %1) {edge_padding_high = dense<1> : tensor<1xi64>, edge_padding_low = dense<0> : tensor<1xi64>, interior_padding = dense<0> : tensor<1xi64>} : (tensor<1xf32>, tensor<f32>) -> tensor<2xf32>
  return %2 : tensor<2xf32>
  // CHECK: mhlo.constant dense<[2.000000e+00, 1.000000e+00]> : tensor<2xf32>
}

// CHECK-LABEL: @pad_complex_fold
func.func @pad_complex_fold() -> tensor<2xcomplex<f32>> {
  %0 = mhlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<1xcomplex<f32>>
  %1 = mhlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
  %2 = "mhlo.pad"(%0, %1) {edge_padding_high = dense<1> : tensor<1xi64>, edge_padding_low = dense<0> : tensor<1xi64>, interior_padding = dense<0> : tensor<1xi64>} : (tensor<1xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2xcomplex<f32>>
  return %2 : tensor<2xcomplex<f32>>
  // CHECK: mhlo.constant dense<[(2.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<2xcomplex<f32>>
}

// CHECK-LABEL: @identity_broadcast_reshape
func.func @identity_broadcast_reshape(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %0 = "mhlo.broadcast"(%arg0) {
    broadcast_sizes = dense<[1]> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<1x128xf32>
  %1 = "mhlo.reshape"(%0) : (tensor<1x128xf32>) -> tensor<128xf32>
  func.return %1 : tensor<128xf32>
  // CHECK: return %arg0 : tensor<128xf32>
}

// CHECK-LABEL: @identity_broadcast_in_dim_reshape
func.func @identity_broadcast_in_dim_reshape(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = dense<[1]> : tensor<1xi64> } : (tensor<128xf32>) -> tensor<1x128xf32>
  %1 = "mhlo.reshape"(%0) : (tensor<1x128xf32>) -> tensor<128xf32>
  func.return %1 : tensor<128xf32>
  // CHECK: return %arg0 : tensor<128xf32>
}

// CHECK-LABEL: @eliminate_identity_convert
func.func @eliminate_identity_convert(%arg : tensor<?x32xi16>) -> tensor<?x32xi16> {
  // CHECK-NOT: mhlo.convert
  %0 = "mhlo.convert"(%arg) : (tensor<?x32xi16>) -> tensor<?x32xi16>
  // CHECK: return %arg0 : tensor<?x32xi16>
  func.return %0 : tensor<?x32xi16>
}

func.func @fold_fptosi() -> tensor<i16> {
  %0 = mhlo.constant dense<65535.000000e+00> : tensor<f32>
  // CHECK: mhlo.constant dense<32767> : tensor<i16>
  %1 = "mhlo.convert"(%0) : (tensor<f32>) -> tensor<i16>
  func.return %1 : tensor<i16>
}

func.func @fold_fptosi_rounding() -> tensor<i16> {
  %0 = mhlo.constant dense<-1.5> : tensor<f32>
  // CHECK: mhlo.constant dense<-1> : tensor<i16>
  %1 = "mhlo.convert"(%0) : (tensor<f32>) -> tensor<i16>
  func.return %1 : tensor<i16>
}

func.func @fold_fptoui() -> tensor<ui16> {
  %0 = mhlo.constant dense<-1.000000e+00> : tensor<f32>
  // CHECK: mhlo.constant dense<0> : tensor<ui16>
  %1 = "mhlo.convert"(%0) : (tensor<f32>) -> tensor<ui16>
  func.return %1 : tensor<ui16>
}

func.func @fold_sitofp() -> tensor<f32> {
  %0 = mhlo.constant dense<-1> : tensor<i16>
  // CHECK: mhlo.constant dense<-1.000000e+00> : tensor<f32>
  %1 = "mhlo.convert"(%0) : (tensor<i16>) -> tensor<f32>
  func.return %1 : tensor<f32>
}

func.func @fold_uitofp() -> tensor<f32> {
  %0 = mhlo.constant dense<65535> : tensor<ui16>
  // CHECK: mhlo.constant dense<6.553500e+04> : tensor<f32>
  %1 = "mhlo.convert"(%0) : (tensor<ui16>) -> tensor<f32>
  func.return %1 : tensor<f32>
}

func.func @fold_uitoui() -> tensor<ui32> {
  %0 = mhlo.constant dense<65535> : tensor<ui16>
  // CHECK: mhlo.constant dense<65535> : tensor<ui32>
  %1 = "mhlo.convert"(%0) : (tensor<ui16>) -> tensor<ui32>
  func.return %1 : tensor<ui32>
}

func.func @fold_uitosi() -> tensor<i32> {
  %0 = mhlo.constant dense<65535> : tensor<ui16>
  // CHECK: mhlo.constant dense<65535> : tensor<i32>
  %1 = "mhlo.convert"(%0) : (tensor<ui16>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

func.func @fold_sitoui() -> tensor<ui32> {
  %0 = mhlo.constant dense<-1> : tensor<i16>
  // CHECK: mhlo.constant dense<4294967295> : tensor<ui32>
  %1 = "mhlo.convert"(%0) : (tensor<i16>) -> tensor<ui32>
  func.return %1 : tensor<ui32>
}

func.func @fold_sitosi() -> tensor<i32> {
  %0 = mhlo.constant dense<-1> : tensor<i16>
  // CHECK: mhlo.constant dense<-1> : tensor<i32>
  %1 = "mhlo.convert"(%0) : (tensor<i16>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK-LABEL: @eliminate_redundant_reshape
func.func @eliminate_redundant_reshape(%arg : tensor<1x32xi16>) -> tensor<1x32xi16> {
  %0 = "mhlo.reshape"(%arg) : (tensor<1x32xi16>) -> tensor<2x16xi16>
  %1 = "mhlo.reshape"(%0) : (tensor<2x16xi16>) -> tensor<1x32xi16>
  // CHECK: return %arg0 : tensor<1x32xi16>
  func.return %1 : tensor<1x32xi16>
}

// CHECK-LABEL: @eliminate_identity_reshape
func.func @eliminate_identity_reshape(%arg : tensor<1x32xi16>) -> tensor<1x32xi16> {
  // CHECK-NOT: mhlo.reshape
  %0 = "mhlo.reshape"(%arg) : (tensor<1x32xi16>) -> tensor<1x32xi16>
  // CHECK: return %arg0 : tensor<1x32xi16>
  func.return %0 : tensor<1x32xi16>
}

// CHECK-LABEL: @broadcast_of_reshape
func.func @broadcast_of_reshape(%arg: tensor<?xf32>,
                           %shape: tensor<2xindex>) -> tensor<?x?xf32> {
  %0 = "mhlo.dynamic_reshape"(%arg, %shape)
      : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%0, %shape) {
    broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  func.return %1 : tensor<?x?xf32>
}
// CHECK: [[RESHAPE:%.*]] = "mhlo.dynamic_reshape"
// CHECK: return [[RESHAPE]]

// CHECK-LABEL: @permutation_broadcast_of_reshape
func.func @permutation_broadcast_of_reshape(%arg: tensor<?xf32>,
    %shape: tensor<2xindex>) -> tensor<?x?xf32> {
  %0 = "mhlo.dynamic_reshape"(%arg, %shape)
      : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%0, %shape) {
    broadcast_dimensions = dense<[1, 0]> : tensor<2xi64>
  } : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  func.return %1 : tensor<?x?xf32>
}
// CHECK: mhlo.dynamic_reshape
// CHECK: mhlo.dynamic_broadcast_in_dim

// CHECK-LABEL: @reshape_of_same_shape_op_result
func.func @reshape_of_same_shape_op_result(%arg: tensor<?xf32>,
    %shape: tensor<2xindex>) -> tensor<?x?xf32> {
  %0 = "mhlo.dynamic_reshape"(%arg, %shape)
    : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %1 = "mhlo.abs"(%0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "mhlo.dynamic_reshape"(%1, %shape)
    : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  func.return %2 : tensor<?x?xf32>
}
// CHECK: mhlo.dynamic_reshape
// CHECK-NEXT: mhlo.abs
// CHECK-NOT: mhlo.dynamic_reshape

// CHECK-LABEL: @map_op_fold
func.func @map_op_fold(%arg: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "mhlo.map"(%arg, %arg1) ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    "mhlo.return"(%b) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK: return %arg1 : tensor<?xf32>

func.func @sort_drop_second_arg(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<3xi32> {
  %0:2 = "mhlo.sort"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>):
    %1 = "mhlo.compare"(%arg2, %arg3) {
      comparison_direction = #mhlo<"comparison_direction GT">
    } : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  }) {dimension = 0 : i64,
      is_stable = false
  } : (tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>)
  func.return %0#0 : tensor<3xi32>
}
// CHECK-LABEL: @sort_drop_second_arg
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         %[[RES:.+]] = "mhlo.sort"(%[[ARG0]])
// CHECK:         ^bb0(%[[ARG2:.+]]: tensor<i32>, %[[ARG3:.+]]: tensor<i32>)
// CHECK:           %[[CMP:.+]] = "mhlo.compare"(%[[ARG2]], %[[ARG3]])
// CHECK-SAME:        {comparison_direction = #mhlo<"comparison_direction GT">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK:           "mhlo.return"(%[[CMP]]) : (tensor<i1>) -> ()
// CHECK:         {dimension = 0 : i64, is_stable = false} : (tensor<3xi32>) -> tensor<3xi32>
// CHECK:         return %[[RES]] : tensor<3xi32>

func.func @sort_no_dim_provided(%arg0: tensor<3x5xi32>) -> tensor<3x5xi32> {
  %0 = "mhlo.sort"(%arg0) ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %1 = "mhlo.compare"(%arg1, %arg2) {
      comparison_direction = #mhlo<"comparison_direction GT">
    } : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  }) {dimension = -1 : i64,
      is_stable = false
  } : (tensor<3x5xi32>) -> tensor<3x5xi32>
  func.return %0 : tensor<3x5xi32>
}
// CHECK-LABEL: @sort_no_dim_provided
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK:         %[[RES:.+]] = "mhlo.sort"(%[[ARG0]])
// CHECK:           dimension = 1 : i64
// CHECK:         return %[[RES]] : tensor<3x5xi32>

// CHECK-LABEL: @reshape_splat_of_bools
func.func public @reshape_splat_of_bools() -> tensor<2x1xi1> {
  // CHECK: mhlo.constant dense<true> : tensor<2x1xi1>
  %0 = mhlo.constant dense<true> : tensor<2xi1>
  %1 = "mhlo.reshape"(%0) : (tensor<2xi1>) -> tensor<2x1xi1>
  return %1 : tensor<2x1xi1>
}
