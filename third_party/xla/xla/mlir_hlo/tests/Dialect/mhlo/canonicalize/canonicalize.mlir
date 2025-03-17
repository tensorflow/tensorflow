// RUN: mlir-hlo-opt %s -pass-pipeline='builtin.module(func.func(canonicalize))' | FileCheck %s

////////
// BroadcastOp(deprecated)

// CHECK-LABEL: func @broadcast_identity
func.func @broadcast_identity(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  // CHECK: return %arg0
  %0 = "mhlo.broadcast"(%arg0) <{broadcast_sizes = dense<[]> : tensor<0xi64>}> : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  func.return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func @broadcast_dynamic_shape_identity
func.func @broadcast_dynamic_shape_identity(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK: return %arg0
  %0 = "mhlo.broadcast"(%arg0) <{broadcast_sizes = dense<[]> : tensor<0xi64>}> : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: func @broadcast_dynamic_shape_not_identity
func.func @broadcast_dynamic_shape_not_identity(%arg0: tensor<?x?x?xf32>) -> tensor<20x?x?x?xf32> {
  // CHECK: mhlo.broadcast
  %0 = "mhlo.broadcast"(%arg0) <{broadcast_sizes = dense<[20]> : tensor<1xi64>}> : (tensor<?x?x?xf32>) -> tensor<20x?x?x?xf32>
  func.return %0 : tensor<20x?x?x?xf32>
}

////////
// BroadcastInDimOp

// CHECK-LABEL: func @broadcast_consecutive
func.func @broadcast_consecutive(%arg0: tensor<2x3xf32>) -> tensor<2x3x4x5xf32> {
  // CHECK: mhlo.broadcast_in_dim
  // CHECK-SAME: broadcast_dimensions = dense<[0, 1]>
  // CHECK-NEXT: return
  %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}> : (tensor<2x3xf32>) -> tensor<2x3x4xf32>
  %1 = "mhlo.broadcast_in_dim"(%0) <{broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>}> : (tensor<2x3x4xf32>) -> tensor<2x3x4x5xf32>
  func.return %1 : tensor<2x3x4x5xf32>
}

// CHECK-LABEL: func @broadcast_in_dim_equivalent_reshape
func.func @broadcast_in_dim_equivalent_reshape(%arg0: tensor<2x3x4xf32>) -> tensor<1x2x3x4xf32> {
  // CHECK: mhlo.reshape
  %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[1, 2, 3]> : tensor<3xi64>}> : (tensor<2x3x4xf32>) -> tensor<1x2x3x4xf32>
  func.return %0 : tensor<1x2x3x4xf32>
}

// CHECK-LABEL: func @broadcast_in_dim_not_identity_because_it_actually_broadcasts
func.func @broadcast_in_dim_not_identity_because_it_actually_broadcasts(%arg0: tensor<1x2xf32>) -> tensor<2x2xf32> {
  // CHECK: mhlo.broadcast_in_dim
  %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}> : (tensor<1x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func @broadcast_in_dim_equivalent_transpose
func.func @broadcast_in_dim_equivalent_transpose(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: mhlo.transpose
  // CHECK-SAME: permutation = dense<[1, 0]>
  %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[1, 0]> : tensor<2xi64>}> : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
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

////////
// ComplexOp

// CHECK-LABEL: @complex_expand
func.func @complex_expand_fold(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = "mhlo.complex"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xcomplex<f32>>)
  %1 = mhlo.real %0 : (tensor<4xcomplex<f32>>) -> (tensor<4xf32>)
  %2 = "mhlo.imag"(%0) : (tensor<4xcomplex<f32>>) -> (tensor<4xf32>)
  // CHECK: return %arg0, %arg1
  func.return %1, %2 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: @complex_collapse
func.func @complex_collapse_fold(%arg0: tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  %0 = mhlo.real %arg0 : (tensor<4xcomplex<f32>>) -> (tensor<4xf32>)
  %1 = "mhlo.imag"(%arg0) : (tensor<4xcomplex<f32>>) -> (tensor<4xf32>)
  %2 = "mhlo.complex"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xcomplex<f32>>
  // CHECK: return %arg0
  func.return %2 : tensor<4xcomplex<f32>>
}

////////
// ConcatenateOp

// CHECK-LABEL: concatenate_noop
func.func @concatenate_noop(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK-SAME: [[ARG:%.+]]: tensor<4xi32>
  %0 = "mhlo.concatenate"(%arg0) <{ dimension = 0 : i64 }> : (tensor<4xi32>) -> tensor<4xi32>

  // CHECK: return [[ARG]]
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: concatenate_remove_operand
func.func @concatenate_remove_operand(%arg0: tensor<4xi32>, %arg1: tensor<0xi32>) -> tensor<4xi32> {
  // CHECK-SAME: [[ARG0:%.+]]: tensor<4xi32>
  // CHECK-SAME: [[ARG1:%.+]]: tensor<0xi32>
  %0 = "mhlo.concatenate"(%arg0, %arg1) <{ dimension = 0 : i64 }> : (tensor<4xi32>, tensor<0xi32>) -> tensor<4xi32>

  // CHECK: return [[ARG0]]
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: concatenate_forward
func.func @concatenate_forward(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<12xi32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) <{ dimension = 0 : i64 }> : (tensor<4xi32>, tensor<4xi32>) -> tensor<8xi32>
  %1 = mhlo.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  // CHECK: "mhlo.concatenate"(%arg0, %arg1, %0) <{dimension = 0 : i64}> : (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<12xi32>
  %2 = "mhlo.concatenate"(%0, %1) <{ dimension = 0 : i64 }> : (tensor<8xi32>, tensor<4xi32>) -> tensor<12xi32>

  func.return %2 : tensor<12xi32>
}

// CHECK-LABEL: concatenate_empty_bool
func.func @concatenate_empty_bool(%arg0: tensor<0xi1>, %arg1: tensor<0xi1>) -> tensor<0xi1> {
  // CHECK: mhlo.constant
  %0 = "mhlo.concatenate"(%arg0, %arg1) <{ dimension = 0 : i64 }> : (tensor<0xi1>, tensor<0xi1>) -> tensor<0xi1>

  func.return %0 : tensor<0xi1>
}

// CHECK-LABEL: concatenate_empty_int
func.func @concatenate_empty_int(%arg0: tensor<0xi32>, %arg1: tensor<0xi32>) -> tensor<0xi32> {
  // CHECK: mhlo.constant
  %0 = "mhlo.concatenate"(%arg0, %arg1) <{ dimension = 0 : i64 }> : (tensor<0xi32>, tensor<0xi32>) -> tensor<0xi32>

  func.return %0 : tensor<0xi32>
}

// CHECK-LABEL: concatenate_empty_float
func.func @concatenate_empty_float(%arg0: tensor<0xf32>, %arg1: tensor<0xf32>) -> tensor<0xf32> {
  // CHECK: mhlo.constant
  %0 = "mhlo.concatenate"(%arg0, %arg1) <{ dimension = 0 : i64 }> : (tensor<0xf32>, tensor<0xf32>) -> tensor<0xf32>

  func.return %0 : tensor<0xf32>
}

// CHECK-LABEL: concatenate_const_1D
func.func @concatenate_const_1D() -> tensor<4xi32> {
  // CHECK: [[VAL:%.+]]= mhlo.constant dense<[0, 1, 2, 3]>
  %0 = mhlo.constant dense<[0, 1]> : tensor<2xi32>
  %1 = mhlo.constant dense<[2, 3]> : tensor<2xi32>
  %2 = "mhlo.concatenate"(%0, %1) <{ dimension = 0 : i64 }> : (tensor<2xi32>, tensor<2xi32>) -> tensor<4xi32>

  // CHECK: return [[VAL]]
  func.return %2 : tensor<4xi32>
}

// CHECK-LABEL: concatenate_const_1D_float
func.func @concatenate_const_1D_float() -> tensor<4xf32> {
  // CHECK: [[VAL:%.+]] = mhlo.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]>

  %0 = mhlo.constant dense<[0.0, 1.0]> : tensor<2xf32>
  %1 = mhlo.constant dense<[2.0, 3.0]> : tensor<2xf32>
  %2 = "mhlo.concatenate"(%0, %1) <{ dimension = 0 : i64 }> : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>

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
  %2 = "mhlo.concatenate"(%0, %1) <{ dimension = 0 : i64 }> : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>

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
  %2 = "mhlo.concatenate"(%0, %1) <{ dimension = 1 : i64 }> : (tensor<2x1xi32>, tensor<2x1xi32>) -> tensor<2x2xi32>

  // CHECK: return [[VAL]]
  func.return %2 : tensor<2x2xi32>
}

////////
// CopyOp

// CHECK-LABEL: func @fold_copy
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @fold_copy(%arg : tensor<1x4xf32>) -> tensor<1x4xf32> {
  // CHECK: return [[ARG]]
  %0 = "mhlo.copy"(%arg) : (tensor<1x4xf32>) -> tensor<1x4xf32>
  func.return %0 : tensor<1x4xf32>
}

////////
// DynamicBroadcastInDimOp

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic
func.func @dynamic_broadcast_in_dim_op_not_actually_dynamic(%arg0: tensor<4xf32>, %arg1: tensor<2xi64>) -> tensor<5x4xf32> {
  // CHECK: %[[RESULT:.+]] = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<1> : tensor<1xi64>}> : (tensor<4xf32>) -> tensor<5x4xf32>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %arg1) <{ broadcast_dimensions = dense<1> : tensor<1xi64> }> : (tensor<4xf32>, tensor<2xi64>) -> tensor<5x4xf32>
  // CHECK: return %[[RESULT]] : tensor<5x4xf32>
  func.return %0 : tensor<5x4xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_shape
func.func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_shape(%arg0: tensor<i32>) -> tensor<4x32xi32> {
  %0 = mhlo.constant dense<[4, 32]> : tensor<2xi32>
  // CHECK: %[[RESULT:.+]] = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<i32>) -> tensor<4x32xi32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %0) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<i32>, tensor<2xi32>) -> tensor<?x32xi32>
  %2 = "mhlo.dynamic_reshape"(%1, %0) : (tensor<?x32xi32>, tensor<2xi32>) -> tensor<4x32xi32>
  // CHECK: return %[[RESULT]] : tensor<4x32xi32>
  func.return %2 : tensor<4x32xi32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_index_shape
func.func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_index_shape(%arg0: tensor<f32>) -> tensor<4x32xf32> {
  %0 = shape.const_shape [4, 32] : tensor<2xindex>
  // CHECK: %[[RESULT:.+]] = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<4x32xf32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %0) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>, tensor<2xindex>) -> tensor<?x32xf32>
  %2 = "mhlo.dynamic_reshape"(%1, %0) : (tensor<?x32xf32>, tensor<2xindex>) -> tensor<4x32xf32>
  // CHECK: return %[[RESULT]] : tensor<4x32xf32>
  func.return %2 : tensor<4x32xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_requires_cast
func.func @dynamic_broadcast_in_dim_op_not_actually_dynamic_constant_requires_cast(%arg0: tensor<f32>) -> tensor<?x?xf32> {
  %0 = shape.const_shape [4, 32] : tensor<2xindex>
  // CHECK: %[[BCAST:.+]] = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<4x32xf32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %0) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK: %[[RESULT:.*]] = tensor.cast %[[BCAST]] : tensor<4x32xf32> to tensor<?x?xf32>
  // CHECK: return %[[RESULT]] : tensor<?x?xf32>
  func.return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_op_almost_not_actually_dynamic
func.func @dynamic_broadcast_in_dim_op_almost_not_actually_dynamic(%arg0: tensor<?xf32>, %arg1: tensor<2xi64>) -> tensor<5x4xf32> {
  // CHECK: %[[RESULT:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg0, %arg1) <{broadcast_dimensions = dense<1> : tensor<1xi64>}> : (tensor<?xf32>, tensor<2xi64>) -> tensor<5x4xf32>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %arg1) <{ broadcast_dimensions = dense<1> : tensor<1xi64> }> : (tensor<?xf32>, tensor<2xi64>) -> tensor<5x4xf32>
  // CHECK: return %[[RESULT]] : tensor<5x4xf32>
  func.return %0 : tensor<5x4xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_to_same_shape_1
func.func @dynamic_broadcast_in_dim_to_same_shape_1(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<?xf32>
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<1xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %0) <{ broadcast_dimensions = dense<0> : tensor<1xi64> }> : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK: return %[[ARG]] : tensor<?xf32>
  func.return %2 : tensor<?xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_to_same_shape_2
func.func @dynamic_broadcast_in_dim_to_same_shape_2(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<?xf32>
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> !shape.shape
  %1 = shape.to_extent_tensor %0 : !shape.shape -> tensor<1xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %1) <{ broadcast_dimensions = dense<0> : tensor<1xi64> }> : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK: return %[[ARG]] : tensor<?xf32>
  func.return %2 : tensor<?xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_to_same_shape_3
func.func @dynamic_broadcast_in_dim_to_same_shape_3(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<?xf32>
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<?xindex>
  %1 = tensor.cast %0 : tensor<?xindex> to tensor<1xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %1) <{ broadcast_dimensions = dense<0> : tensor<1xi64> }> : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK: return %[[ARG]] : tensor<?xf32>
  func.return %2 : tensor<?xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_to_same_shape_4
func.func @dynamic_broadcast_in_dim_to_same_shape_4(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<?xf32>
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> !shape.shape
  %1 = shape.to_extent_tensor %0 : !shape.shape -> tensor<?xindex>
  %2 = tensor.cast %1 : tensor<?xindex> to tensor<1xindex>
  %3 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %2) <{ broadcast_dimensions = dense<0> : tensor<1xi64> }> : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK: return %[[ARG]] : tensor<?xf32>
  func.return %3 : tensor<?xf32>
}

// CHECK-LABEL: func @dynamic_broadcast_in_dim_all_dims_non_expanding
func.func @dynamic_broadcast_in_dim_all_dims_non_expanding(%arg0: tensor<?xf32>, %arg1: tensor<1xindex>) -> tensor<?xf32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<?xf32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %arg1) {
    broadcast_dimensions = dense<0> : tensor<1xi64>,
    known_expanding_dimensions = dense<> : tensor<0xi64>,
    known_nonexpanding_dimensions = dense<0> : tensor<1xi64>
  } : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK: return %[[ARG]] : tensor<?xf32>
  func.return %1 : tensor<?xf32>
}

// CHECK-LABEL: @broadcast_of_reshape
func.func @broadcast_of_reshape(%arg: tensor<?xf32>,
                           %shape: tensor<2xindex>) -> tensor<?x?xf32> {
  // CHECK: [[RESHAPE:%.*]] = mhlo.dynamic_reshape
  // CHECK: return [[RESHAPE]]
  %0 = "mhlo.dynamic_reshape"(%arg, %shape) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%0, %shape) { broadcast_dimensions = dense<[0, 1]> : tensor<2xi64> } : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  func.return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: @permutation_broadcast_of_reshape
func.func @permutation_broadcast_of_reshape(%arg: tensor<?xf32>,
    %shape: tensor<2xindex>) -> tensor<?x?xf32> {
  // CHECK: mhlo.dynamic_reshape
  // CHECK: mhlo.dynamic_broadcast_in_dim
  %0 = "mhlo.dynamic_reshape"(%arg, %shape) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%0, %shape) { broadcast_dimensions = dense<[1, 0]> : tensor<2xi64> } : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  func.return %1 : tensor<?x?xf32>
}

////////
// DynamicGatherOp

// CHECK-LABEL: @simplify_dynamic_gather_i64
func.func @simplify_dynamic_gather_i64(%arg0: tensor<375682x256xf16>, %arg1: tensor<16x64xi64>) -> tensor<16x64x256xf16> {
  %0 = "arith.constant"() {value = dense<[1, 256]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "mhlo.dynamic_gather"(%arg0, %arg1, %0) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false} : (tensor<375682x256xf16>, tensor<16x64xi64>, tensor<2xi64>) -> tensor<16x64x256xf16>
  // CHECK: %[[RET:.+]] = "mhlo.gather"(%arg0, %arg1) <{dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[1, 256]> : tensor<2xi64>}> : (tensor<375682x256xf16>, tensor<16x64xi64>) -> tensor<16x64x256xf16>
  // CHECK: return %[[RET]]
  return %1 : tensor<16x64x256xf16>
}

// CHECK-LABEL: @simplify_dynamic_gather_i32
func.func @simplify_dynamic_gather_i32(%arg0: tensor<375682x256xf16>, %arg1: tensor<16x64xi64>) -> tensor<16x64x256xf16> {
  %0 = "arith.constant"() {value = dense<[1, 256]> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "mhlo.dynamic_gather"(%arg0, %arg1, %0) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false} : (tensor<375682x256xf16>, tensor<16x64xi64>, tensor<2xi32>) -> tensor<16x64x256xf16>
  // CHECK: %[[RET:.+]] = "mhlo.gather"(%arg0, %arg1) <{dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[1, 256]> : tensor<2xi64>}> : (tensor<375682x256xf16>, tensor<16x64xi64>) -> tensor<16x64x256xf16>
  // CHECK: return %[[RET]]
  return %1 : tensor<16x64x256xf16>
}

////////
// DynamicIotaOp

// CHECK-LABEL: @dynamic_iota_is_static
func.func @dynamic_iota_is_static(%arg0 : tensor<1xindex>) -> tensor<4xi32> {
  // CHECK: [[RESULT:%.*]] = "mhlo.iota"
  // CHECK: return [[RESULT]]
  %0 = "mhlo.dynamic_iota"(%arg0) <{iota_dimension = 0 : i64}> : (tensor<1xindex>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: @dynamic_iota_broadcast
func.func @dynamic_iota_broadcast(%arg0 : tensor<2xindex>) -> tensor<5x?xi32> {
  // CHECK: [[IOTA:%.+]] = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<5xi32>
  // CHECK: [[BROADCAST:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[IOTA]], %arg0) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<5xi32>, tensor<2xindex>) -> tensor<5x?xi32>
  %0 = "mhlo.dynamic_iota"(%arg0) <{iota_dimension = 0 : i64}> : (tensor<2xindex>) -> tensor<5x?xi32>

  // CHECK: return [[BROADCAST]]
  func.return %0 : tensor<5x?xi32>
}

// CHECK-LABEL: @dynamic_iota_broadcast_second
func.func @dynamic_iota_broadcast_second(%arg0 : tensor<2xindex>) -> tensor<5x?xi32> {
  // CHECK-NEXT: [[CAST1:%.+]] = arith.index_cast %arg0 : tensor<2xindex> to tensor<2xi64>
  // CHECK-NEXT: [[SLICE:%.+]] = "mhlo.slice"([[CAST1]]) <{limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK-NEXT: [[CAST2:%.+]] = arith.index_cast [[SLICE]] : tensor<1xi64> to tensor<1xindex>
  // CHECK-NEXT: [[IOTA:%.+]] = "mhlo.dynamic_iota"([[CAST2]]) <{iota_dimension = 0 : i64}> : (tensor<1xindex>) -> tensor<?xi32>
  // CHECK-NEXT: [[BROADCAST:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[IOTA]], %arg0) <{broadcast_dimensions = dense<1> : tensor<1xi64>}> : (tensor<?xi32>, tensor<2xindex>) -> tensor<5x?xi32>
  %0 = "mhlo.dynamic_iota"(%arg0) <{iota_dimension = 1 : i64}> : (tensor<2xindex>) -> tensor<5x?xi32>

  // CHECK: return [[BROADCAST]]
  func.return %0 : tensor<5x?xi32>
}

// CHECK-LABEL: @dynamic_iota_constant
func.func @dynamic_iota_constant(%arg0 : tensor<2xindex>) -> tensor<1x?xi32> {
  // CHECK: [[IOTA:%.+]] = mhlo.constant dense<0> : tensor<1xi32>
  // CHECK: [[BROADCAST:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[IOTA]], %arg0) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<1xi32>, tensor<2xindex>) -> tensor<1x?xi32>
  %0 = "mhlo.dynamic_iota"(%arg0) <{iota_dimension = 0 : i64}> : (tensor<2xindex>) -> tensor<1x?xi32>

  // CHECK: return [[BROADCAST]]
  func.return %0 : tensor<1x?xi32>
}

////////
// DynamicReshapeOp

// CHECK-LABEL: func @dynamic_reshape_not_actually_dynamic
func.func @dynamic_reshape_not_actually_dynamic(%arg0: tensor<4xf32>, %shape: tensor<2xindex>) -> tensor<4x1xf32> {
  // CHECK: mhlo.reshape
  %0 = "mhlo.dynamic_reshape"(%arg0, %shape) : (tensor<4xf32>, tensor<2xindex>) -> tensor<4x1xf32>
  func.return %0 : tensor<4x1xf32>
}

// CHECK-LABEL: func @shape_of_dynamic_reshape
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]
func.func @shape_of_dynamic_reshape(%arg0: tensor<?xf32>, %shape: tensor<2xindex>) -> tensor<2xindex> {
  // CHECK: return [[ARG1]]
  %0 = "mhlo.dynamic_reshape"(%arg0, %shape) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %1 = shape.shape_of %0 : tensor<?x?xf32> -> tensor<2xindex>
  func.return %1 : tensor<2xindex>
}

// CHECK-LABEL: func @dynamic_reshape_rank_1_to_rank_1
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]
func.func @dynamic_reshape_rank_1_to_rank_1(%arg0: tensor<?xcomplex<f32>>,
    %shape: tensor<?xindex>) -> tensor<?xf32> {
  // CHECK: [[RES:%[a-zA-Z0-9]+]] = mhlo.real [[ARG0]] : (tensor<?xcomplex<f32>>) -> tensor<?xf32>
  // CHECK: return [[RES]]
  %0 = mhlo.real %arg0: (tensor<?xcomplex<f32>>) -> tensor<?xf32>
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
  %0 = "mhlo.dynamic_reshape"(%arg0, %shape) : (tensor<?xf16>, tensor<?xindex>) -> tensor<?xf16>
  %1 = shape.shape_of %0 : tensor<?xf16> -> tensor<?xindex>
  %2 = shape.num_elements %1 : tensor<?xindex> -> index
  %3 = tensor.from_elements %2 : tensor<1xindex>
  %4 = "mhlo.dynamic_reshape"(%0, %3) : (tensor<?xf16>, tensor<1xindex>) -> tensor<?xf16>
  func.return %4 : tensor<?xf16>
}

////////
// GatherOp

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
  // CHECK:  %[[RET:.*]] = "mhlo.slice"(%arg0) <{limit_indices = dense<[4, 6, 7]> : tensor<3xi64>, start_indices = dense<[1, 0, 2]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}> : (tensor<5x6x7xf32>) -> tensor<3x6x5xf32>
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
  // CHECK:  %[[RET:.*]] = "mhlo.slice"(%arg0) <{limit_indices = dense<[5, 6, 5]> : tensor<3xi64>, start_indices = dense<[0, 0, 1]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}> : (tensor<5x6x7xf32>) -> tensor<5x6x4xf32>
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
  // CHECK:  %[[V0:.*]] = "mhlo.slice"(%arg0) <{limit_indices = dense<[4, 6, 3]> : tensor<3xi64>, start_indices = dense<[1, 0, 2]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>}> : (tensor<5x6x7xf32>) -> tensor<3x6x1xf32>
  // CHECK:  %[[V1:.*]] = mhlo.reshape %[[V0]] : (tensor<3x6x1xf32>) -> tensor<3x6xf32>
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
  // CHECK:  %[[V0:.*]] = "mhlo.slice"(%arg0) <{limit_indices = dense<[4, 2]> : tensor<2xi64>, start_indices = dense<[3, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}> : (tensor<4x2xui32>) -> tensor<1x2xui32>
  // CHECK:  %[[V1:.*]] = mhlo.reshape %[[V0]] : (tensor<1x2xui32>) -> tensor<2xui32>
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
  // CHECK:  %[[V0:.*]] = "mhlo.slice"(%arg0) <{limit_indices = dense<[1, 2]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}> : (tensor<4x2xui32>) -> tensor<1x2xui32>
  // CHECK:  %[[V1:.*]] = mhlo.reshape %[[V0]] : (tensor<1x2xui32>) -> tensor<2xui32>
  // CHECK: return %[[V1]] : tensor<2xui32>
}

////////
// IotaOp

// CHECK-LABEL: @iota_constant
func.func @iota_constant() -> tensor<1xi32> {
  // CHECK: [[CONST:%.+]] = mhlo.constant dense<0> : tensor<1xi32>
  %0 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<1xi32>

  // CHECK: return [[CONST]] : tensor<1xi32>
  func.return %0 : tensor<1xi32>
}

// CHECK-LABEL: @iota_constant_multi
func.func @iota_constant_multi() -> tensor<1x4xi32> {
  // CHECK: [[CONST:%.+]] = mhlo.constant dense<0> : tensor<1x4xi32>
  %0 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<1x4xi32>

  // CHECK: return [[CONST]] : tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// CHECK-LABEL: @iota_not_lowered_to_constant
func.func @iota_not_lowered_to_constant() -> tensor<4xi32> {
  // CHECK: [[RESULT:%.*]] = "mhlo.iota"
  // CHECK: return [[RESULT]]
  %0 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: @iota_broadcast
func.func @iota_broadcast() -> tensor<5x4xi32> {
  // CHECK: [[IOTA:%.+]] = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<5xi32>
  // CHECK: [[RESULT:%.+]] = "mhlo.broadcast_in_dim"([[IOTA]]) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<5xi32>) -> tensor<5x4xi32>
  %0 = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<5x4xi32>

  func.return %0 : tensor<5x4xi32>
}

// CHECK-LABEL: @iota_broadcast
func.func @iota_broadcast_second() -> tensor<5x4xi32> {
  // CHECK: [[IOTA:%.+]] = "mhlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4xi32>
  // CHECK: [[RESULT:%.+]] = "mhlo.broadcast_in_dim"([[IOTA]]) <{broadcast_dimensions = dense<1> : tensor<1xi64>}> : (tensor<4xi32>) -> tensor<5x4xi32>
  %0 = "mhlo.iota"() <{iota_dimension = 1 : i64}> : () -> tensor<5x4xi32>

  func.return %0 : tensor<5x4xi32>
}

////////
// PadOp

// CHECK-LABEL: @pad_zero_length
func.func @pad_zero_length(%arg0: tensor<5x0xf32>, %arg1: tensor<f32>) -> tensor<7x2xf32> {
  // CHECK: %[[RES:.+]] = "mhlo.broadcast_in_dim"(%arg1) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<7x2xf32>
  %0 = "mhlo.pad"(%arg0, %arg1) {
    edge_padding_low = dense<1> : tensor<2xi64>,
    edge_padding_high = dense<1> : tensor<2xi64>,
    interior_padding = dense<0> : tensor<2xi64>
  } : (tensor<5x0xf32>, tensor<f32>) -> tensor<7x2xf32>
  // CHECK: return %[[RES]]
  func.return %0 : tensor<7x2xf32>
}

////////
// DynamicSliceOp

// CHECK-LABEL: @fold_dynamic_slice
func.func @fold_dynamic_slice(%767: tensor<i32>, %203: tensor<i32>) -> tensor<1x1xi32> {
  %28 = mhlo.constant dense<256> : tensor<6x1xi32>
  %769 = "mhlo.dynamic_slice"(%28, %767, %203) <{slice_sizes = dense<1> : tensor<2xi64>}> : (tensor<6x1xi32>, tensor<i32>, tensor<i32>) -> tensor<1x1xi32>

  // CHECK: %[[RESULT:.*]] = mhlo.constant dense<256>
  // CHECK: return %[[RESULT]]
  return %769 : tensor<1x1xi32>
}

////////
// RealDynamicSliceOp

// CHECK-LABEL: @simplify_real_dynamic_slice_to_slice
func.func @simplify_real_dynamic_slice_to_slice(%arg0: tensor<?x4xf32>) -> tensor<1x4xf32> {
  %0 = mhlo.constant dense<[0, 0]> : tensor<2xi32>
  %1 = mhlo.constant dense<[1, 4]> : tensor<2xi32>
  %2 = mhlo.constant dense<[1, 1]> : tensor<2xi32>
  %3 = mhlo.real_dynamic_slice %arg0, %0, %1, %2 : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x4xf32>
  // CHECK: %[[RESULT:.*]] =  "mhlo.slice"(%arg0)
  // CHECK-DAG-SAME: start_indices = dense<[0, 0]> : tensor<2xi64>
  // CHECK-DAG-SAME: limit_indices = dense<[1, 4]> : tensor<2xi64>
  // CHECK-DAG-SAME: strides = dense<[1, 1]> : tensor<2xi64>}
  // CHECK: return %[[RESULT]] : tensor<1x4xf32>
  return %3 : tensor<1x4xf32>
}

// CHECK-LABEL: @simplify_real_dynamic_slice_to_dynamic_slice
func.func @simplify_real_dynamic_slice_to_dynamic_slice(%arg0: tensor<?x4xf32>, %arg1: tensor<2xi32>) -> tensor<1x4xf32> {
  %0 = mhlo.constant dense<[1, 4]> : tensor<2xi32>
  %1 = mhlo.add %arg1, %0 : tensor<2xi32>
  %2 = mhlo.constant dense<[1, 1]> : tensor<2xi32>
  %3 = mhlo.real_dynamic_slice %arg0, %arg1, %1, %2 : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x4xf32>
  return %3 : tensor<1x4xf32>
  //      CHECK: [[START_INDEX_0_1D:%.*]] = "mhlo.slice"(%arg1) <{limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK-NEXT: [[START_INDEX_0_0D:%.*]] = mhlo.reshape [[START_INDEX_0_1D]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: [[START_INDEX_1_1D:%.*]] = "mhlo.slice"(%arg1) <{limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK-NEXT: [[START_INDEX_1_0D:%.*]] = mhlo.reshape [[START_INDEX_1_1D]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: [[RESULT:%.*]] = "mhlo.dynamic_slice"(%arg0, [[START_INDEX_0_0D]], [[START_INDEX_1_0D]]) <{
  // CHECK-SAME:   slice_sizes = dense<[1, 4]> : tensor<2xi64>
  // CHECK-SAME: }> : (tensor<?x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x4xf32>
}

////////
// ReshapeOp

// CHECK-LABEL: @reshape_of_same_shape_op_result
func.func @reshape_of_same_shape_op_result(%arg: tensor<?xf32>,
    %shape: tensor<2xindex>) -> tensor<?x?xf32> {
  // CHECK: mhlo.dynamic_reshape
  // CHECK-NEXT: mhlo.abs
  // CHECK-NOT: mhlo.dynamic_reshape
  %0 = "mhlo.dynamic_reshape"(%arg, %shape) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %1 = "mhlo.abs"(%0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "mhlo.dynamic_reshape"(%1, %shape) : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  func.return %2 : tensor<?x?xf32>
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

////////
// SelectOp

// CHECK-LABEL: func @simplify_not_as_select_pred(
func.func @simplify_not_as_select_pred(%arg0 : tensor<4xi1>, %arg1 : tensor<4xf32>, %arg2 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = "mhlo.not"(%arg0) : (tensor<4xi1>) -> tensor<4xi1>
  %1 = "mhlo.select"(%0, %arg1, %arg2) : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>

  // CHECK: %[[R:.*]] = mhlo.select %arg0, %arg2, %arg1
  // CHECK: return %[[R]]
}

// CHECK-LABEL: func @simplify_broadcasted_not_as_select_pred(
func.func @simplify_broadcasted_not_as_select_pred(%arg0 : tensor<1xi1>, %arg1 : tensor<4xf32>, %arg2 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = "mhlo.not"(%arg0) : (tensor<1xi1>) -> tensor<1xi1>
  %1 = "mhlo.broadcast_in_dim"(%0) <{broadcast_dimensions = dense<[0]> : tensor<1xi64> }> : (tensor<1xi1>) -> tensor<4xi1>
  %2 = "mhlo.select"(%1, %arg1, %arg2) : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %2 : tensor<4xf32>

  // CHECK: %[[B:.*]] = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<1xi1>) -> tensor<4xi1>
  // CHECK: %[[R:.*]] = mhlo.select %[[B]], %arg2, %arg1
  // CHECK: return %[[R]]
}

////////
// SetDimensionSizeOp

// CHECK-LABEL: set_dimension_size_is_not_folded
func.func @set_dimension_size_is_not_folded() -> tensor<?x1xi64, #mhlo.type_extensions<bounds = [5, ?]>> {
    // CHECK: set_dimension_size
    %0 = "mhlo.constant"() {value = dense<[[1],[2],[3],[4],[5],[6],[7]]> : tensor<7x1xi64>} : () -> tensor<7x1xi64>
    %size = "mhlo.constant"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
    %1 = "mhlo.set_dimension_size"(%0, %size) <{dimension = 0 : i64}> : (tensor<7x1xi64>, tensor<i32>) -> tensor<?x1xi64, #mhlo.type_extensions<bounds = [5, ?]>>
    return %1 : tensor<?x1xi64, #mhlo.type_extensions<bounds = [5, ?]>>
}

////////
// SliceOp

// CHECK-LABEL: dynamic_update_slice_fold_length_0
func.func @dynamic_update_slice_fold_length_0(%arg0: tensor<3x4xi64>, %arg1: tensor<3x0xi64>) -> tensor<3x4xi64> {
  // CHECK: return %arg0
  %0 = mhlo.constant dense<0> : tensor<i64>
  %1 = "mhlo.dynamic_update_slice"(%arg0, %arg1, %0, %0) : (tensor<3x4xi64>, tensor<3x0xi64>, tensor<i64>, tensor<i64>) -> tensor<3x4xi64>
  func.return %1 : tensor<3x4xi64>
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
  // CHECK: %[[VAL:.*]] = mhlo.dynamic_update_slice %arg0, %arg1, %[[CST]], %[[CST]] : (tensor<?x?xi64>, tensor<?x?xi64>, tensor<i64>, tensor<i64>) -> tensor<?x?xi64>
  // CHECK: return %[[VAL]] : tensor<?x?xi64>
}

// CHECK-LABEL: dynamic_slice_variable_start
func.func @dynamic_slice_variable_start(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  // CHECK: "mhlo.dynamic_slice"
  %1 = "mhlo.dynamic_slice"(%arg0, %arg1, %arg2) <{slice_sizes = dense<[1, 4]> : tensor<2xi64>}> : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
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
  %1 = "mhlo.dynamic_slice"(%arg0, %0) <{slice_sizes = dense<2> : tensor<1xi64>}> : (tensor<4xi32>, tensor<i64>) -> tensor<2xi32>
  func.return %1 : tensor<2xi32>
}

// CHECK-LABEL: dynamic_slice_constant_start_dynamic_shape
func.func @dynamic_slice_constant_start_dynamic_shape(%arg0: tensor<?x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  // CHECK: mhlo.dynamic_slice
  // CHECK-NOT: mhlo.slice
  %0 = mhlo.constant dense<1> : tensor<i64>
  %1 = mhlo.constant dense<0> : tensor<i64>
  %2 = "mhlo.dynamic_slice"(%arg0, %0, %1) <{slice_sizes = dense<[1, 4]> : tensor<2xi64>}> : (tensor<?x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
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
  %2 = "mhlo.dynamic_slice"(%arg0, %0, %1) <{slice_sizes = dense<[1, 4]> : tensor<2xi64>}> : (tensor<8x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
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
  %2 = "mhlo.dynamic_slice"(%arg0, %0, %1) <{slice_sizes = dense<[1, 4]> : tensor<2xi64>}> : (tensor<8x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %2 : tensor<1x4xi32>
}

// CHECK-LABEL: slice_2D_noop
// CHECK-SAME: [[ARG:%.+]]: tensor<2x2xi64>
func.func @slice_2D_noop(%arg0: tensor<2x2xi64>) -> tensor<2x2xi64> {
  %0 = "mhlo.slice"(%arg0) <{ limit_indices = dense<[2, 2]> : tensor<2xi64>, start_indices = dense<[0, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}> : (tensor<2x2xi64>) -> (tensor<2x2xi64>)

  // CHECK-NEXT: return [[ARG]]
  func.return %0 : tensor<2x2xi64>
}

// CHECK-LABEL: slice_concat_empty
func.func @slice_concat_empty(%arg0: tensor<1x5xf32>, %arg1: tensor<1x5xf32>, %arg2: tensor<1x5xf32>) -> tensor<1x5xf32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) <{ dimension = 0 : i64 }> : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<2x5xf32>
  %1 = "mhlo.slice"(%0) <{ limit_indices = dense<[2, 5]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}> : (tensor<2x5xf32>) -> (tensor<0x5xf32>)
  %2 = "mhlo.concatenate"(%1, %arg2) <{ dimension = 0 : i64 }> : (tensor<0x5xf32>, tensor<1x5xf32>) -> tensor<1x5xf32>

  // CHECK: return %arg2
  func.return %2 : tensor<1x5xf32>
}

////////
// SortOp

// CHECK-LABEL: @sort_drop_second_arg
func.func @sort_drop_second_arg(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<3xi32> {
  // CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
  // CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
  // CHECK:         %[[RES:.+]] = "mhlo.sort"(%[[ARG0]]) <{dimension = 0 : i64, is_stable = false}> ({
  // CHECK:         ^bb0(%[[ARG2:.+]]: tensor<i32>, %[[ARG3:.+]]: tensor<i32>)
  // CHECK:           %[[CMP:.+]] = mhlo.compare GT, %[[ARG2]], %[[ARG3]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK:           mhlo.return %[[CMP]] : tensor<i1>
  // CHECK:         }) : (tensor<3xi32>) -> tensor<3xi32>
  // CHECK:         return %[[RES]] : tensor<3xi32>
  %0:2 = "mhlo.sort"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>):
    %1 = "mhlo.compare"(%arg2, %arg3) {
      comparison_direction = #mhlo<comparison_direction GT>
    } : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  }) {dimension = 0 : i64,
      is_stable = false
  } : (tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>)
  func.return %0#0 : tensor<3xi32>
}


// CHECK-LABEL: @sort_no_dim_provided
func.func @sort_no_dim_provided(%arg0: tensor<3x5xi32>) -> tensor<3x5xi32> {
  // CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
  // CHECK:         %[[RES:.+]] = "mhlo.sort"(%[[ARG0]])
  // CHECK:           dimension = 1 : i64
  // CHECK:         return %[[RES]] : tensor<3x5xi32>
  %0 = "mhlo.sort"(%arg0) ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %1 = "mhlo.compare"(%arg1, %arg2) {
      comparison_direction = #mhlo<comparison_direction GT>
    } : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  }) {dimension = -1 : i64,
      is_stable = false
  } : (tensor<3x5xi32>) -> tensor<3x5xi32>
  func.return %0 : tensor<3x5xi32>
}

////////
// TupleOp

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

////////
// WhileOp DCE

// CHECK-LABEL: do_not_dce_while_with_outfeed
func.func @do_not_dce_while_with_outfeed(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK: mhlo.while
  %0 = "mhlo.while"(%arg0) ({
  ^bb0(%arg1: tensor<i64>):
    %1 = "mhlo.compare"(%arg1, %arg1) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i64>, tensor<i64>) -> tensor<i1>
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
    %1 = "mhlo.compare"(%arg1, %arg1) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i64>):
    %1 = "mhlo.create_token"() : () -> !mhlo.token
    "mhlo.return"(%arg1) : (tensor<i64>) -> ()
  }) : (tensor<i64>) -> tensor<i64>

  func.return %arg0 : tensor<i64>
}

// CHECK-LABEL: while_op_dce_no_side_effect
func.func @while_op_dce_no_side_effect(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = mhlo.constant dense<1> : tensor<i32>
  %1 = mhlo.constant dense<10> : tensor<i32>
  %2 = mhlo.constant dense<0> : tensor<i32>
  %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %4 = "mhlo.broadcast_in_dim"(%3) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<10xf32>
  // CHECK: mhlo.while(%iterArg = %2, %iterArg_0 = %3) : tensor<i32>, tensor<10xf32> attributes {mhlo.frontend_attributes = {test_attr = "true"}}
  %5:3 = mhlo.while(%iterArg = %arg0, %iterArg_0 = %2, %iterArg_1 = %4) : tensor<10xf32>, tensor<i32>, tensor<10xf32> attributes {mhlo.frontend_attributes = {test_attr = "true"}}
    cond {
    %6 = mhlo.compare  LT, %iterArg_0, %1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    mhlo.return %6 : tensor<i1>
  } do {
    %6 = "mhlo.dynamic_slice"(%iterArg, %iterArg_0) <{slice_sizes = dense<1> : tensor<1xi64>}> : (tensor<10xf32>, tensor<i32>) -> tensor<1xf32>
    %7 = mhlo.reshape %6 : (tensor<1xf32>) -> tensor<f32>
    %8 = mhlo.sine %7 : tensor<f32>
    %9 = "mhlo.broadcast_in_dim"(%8) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<1xf32>
    %10 = mhlo.dynamic_update_slice %iterArg_1, %9, %iterArg_0 : (tensor<10xf32>, tensor<1xf32>, tensor<i32>) -> tensor<10xf32>
    %11 = mhlo.add %iterArg_0, %0 : tensor<i32>
    mhlo.return %iterArg, %11, %10 : tensor<10xf32>, tensor<i32>, tensor<10xf32>
  }
  return %5#2 : tensor<10xf32>
}

////////
// Tensor/Shape canonicalize

// CHECK-LABEL: concatenate_noop_typecast
func.func @concatenate_noop_typecast(%arg0: tensor<?xi32>) -> tensor<4xi32> {
  // CHECK-SAME: [[ARG:%.+]]: tensor<?xi32>
  // CHECK-NEXT: [[RES:%.+]] = tensor.cast [[ARG]] : tensor<?xi32> to tensor<4xi32>
  %0 = "mhlo.concatenate"(%arg0) <{ dimension = 0 : i64 }> : (tensor<?xi32>) -> tensor<4xi32>

  // CHECK: return [[RES]]
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: slice_unknown_shape
func.func @slice_unknown_shape(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: "mhlo.slice"(%arg0) <{limit_indices = dense<[1, 4]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}> : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %0 = "mhlo.slice"(%arg0) <{limit_indices = dense<[1, 4]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}> : (tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @pad_zero_length_dyn
func.func @pad_zero_length_dyn(%arg0: tensor<?x0xf32>, %arg1: tensor<f32>) -> tensor<?x2xf32> {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  // CHECK-DAG: %[[DIM:.+]] = tensor.dim %arg0, %[[C0]] : tensor<?x0xf32>
  // CHECK-DAG: %[[SUB:.+]] = arith.subi %[[DIM]], %[[C1]]
  // CHECK-DAG: %[[MAX:.+]] = arith.maxsi %[[SUB]], %[[C0]]
  // CHECK-DAG: %[[MUL:.+]] = arith.muli %[[MAX]], %[[C2]]
  // CHECK-DAG: %[[ADD1:.+]] = arith.addi %[[DIM]], %[[MUL]]
  // CHECK-DAG: %[[ADD2:.+]] = arith.addi %[[ADD1]], %[[C2]]
  // CHECK-DAG: %[[SHAPE:.+]] = tensor.from_elements %[[ADD2]], %[[C2]] : tensor<2xindex>
  // CHECK-DAG: %[[BROAD:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg1, %[[SHAPE]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>, tensor<2xindex>) -> tensor<?x2xf32>
  %0 = "mhlo.pad"(%arg0, %arg1) {
    edge_padding_low = dense<1> : tensor<2xi64>,
    edge_padding_high = dense<1> : tensor<2xi64>,
    interior_padding = dense<2> : tensor<2xi64>
  } : (tensor<?x0xf32>, tensor<f32>) -> tensor<?x2xf32>
  // CHECK: return %[[BROAD]]
  func.return %0 : tensor<?x2xf32>
}

// CHECK-LABEL: @dynamic_pad_length_dyn
func.func @dynamic_pad_length_dyn(
  %arg0: tensor<?x0xf32>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>,
  %arg3: tensor<2xi32>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i32
  // CHECK-DAG: %[[CI0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[CI1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[CST:.+]] = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[DIM0:.+]] = tensor.dim %arg0, %[[CI0]]
  // CHECK: %[[CAST:.+]] = arith.index_cast %[[DIM0]] : index to i32
  // CHECK: %[[EX0:.+]] = tensor.extract %arg1[%[[CI0]]]
  // CHECK: %[[EX1:.+]] = tensor.extract %arg2[%[[CI0]]]
  // CHECK: %[[EX2:.+]] = tensor.extract %arg3[%[[CI0]]]
  // CHECK: %[[CMP:.+]] = arith.cmpi slt, %[[CAST]], %[[C1]] : i32
  // CHECK: %[[SUB:.+]] = arith.subi %[[CAST]], %[[C1]] : i32
  // CHECK: %[[SEL:.+]] = arith.select %[[CMP]], %[[C0]], %[[SUB]] : i32
  // CHECK: %[[MUL:.+]] = arith.muli %[[EX2]], %[[SEL]] : i32
  // CHECK: %[[ADD0:.+]] = arith.addi %[[MUL]], %[[CAST]] : i32
  // CHECK: %[[ADD1:.+]] = arith.addi %[[ADD0]], %[[EX0]] : i32
  // CHECK: %[[ADD2:.+]] = arith.addi %[[ADD1]], %[[EX1]] : i32
  // CHECK: %[[EX3:.+]] = tensor.extract %arg1[%[[CI1]]]
  // CHECK: %[[EX4:.+]] = tensor.extract %arg2[%[[CI1]]]
  // CHECK: %[[ADD3:.+]] = arith.addi %[[EX3]], %[[EX4]] : i32
  // CHECK: %[[SHAPE:.+]] = tensor.from_elements %[[ADD2]], %[[ADD3]] : tensor<2xi32>
  // CHECK: %[[BROAD:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[CST]], %[[SHAPE]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}>
  %0 = arith.constant dense<0.0> : tensor<f32>
  %1 = "mhlo.dynamic_pad"(%arg0, %0, %arg1, %arg2, %arg3) {
  } : (tensor<?x0xf32>, tensor<f32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xf32>
  // CHECK: return %[[BROAD]]
  func.return %1 : tensor<?x?xf32>
}
