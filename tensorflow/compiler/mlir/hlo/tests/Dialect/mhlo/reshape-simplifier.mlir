// RUN: mlir-hlo-opt %s -split-input-file -reshape-simplifier | FileCheck %s

// CHECK-LABEL: func @reshape_expand_front
func @reshape_expand_front(%arg0: tensor<?x?xf32>) -> tensor<1x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %shape = tensor.from_elements %c1, %d0, %d1 : tensor<3xindex>
  %reshape = "mhlo.dynamic_reshape"(%arg0, %shape) : (tensor<?x?xf32>, tensor<3xindex>) -> tensor<1x?x?xf32>
// CHECK: tensor.expand_shape %arg0 [
// CHECK-SAME: [0, 1], [2]] : tensor<?x?xf32> into tensor<1x?x?xf32>
  return %reshape : tensor<1x?x?xf32>
}

// -----

// CHECK-LABEL: func @reshape_expand_back
func @reshape_expand_back(%arg0: tensor<?x?xf32>) -> tensor<?x?x1x1xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %shape = tensor.from_elements %d0, %d1, %c1, %c1 : tensor<4xindex>
  %reshape = "mhlo.dynamic_reshape"(%arg0, %shape) : (tensor<?x?xf32>, tensor<4xindex>) -> tensor<?x?x1x1xf32>
// CHECK: tensor.expand_shape %arg0 [
// CHECK-SAME: [0], [1, 2, 3]] : tensor<?x?xf32> into tensor<?x?x1x1xf32>
  return %reshape : tensor<?x?x1x1xf32>
}

// -----

// CHECK-LABEL: func @reshape_undefined
func @reshape_undefined(%arg0: tensor<?xf32>) -> tensor<1x1x1xf32> {
  %c1 = arith.constant 1 : index
  %shape = tensor.from_elements %c1, %c1, %c1 : tensor<3xindex>
  %reshape = "mhlo.dynamic_reshape"(%arg0, %shape) : (tensor<?xf32>, tensor<3xindex>) -> tensor<1x1x1xf32>
// CHECK: mhlo.dynamic_reshape
  return %reshape : tensor<1x1x1xf32>
}

// -----

// CHECK-LABEL: func @compute_reshape_shape
func @compute_reshape_shape(%arg0: tensor<?x?xf32>, %arg1: index) -> tensor<2xi32> {
  %shape = shape.shape_of %arg0: tensor<?x?xf32> -> tensor<2xindex>
  %casted = arith.index_cast %shape : tensor<2xindex> to tensor<2xi32>
  %mul = mhlo.multiply %casted, %casted : tensor<2xi32>
// CHECK:  %[[MUL:.*]] = mhlo.multiply
  %crs = mhlo.compute_reshape_shape %arg1, %mul : index, tensor<2xi32> -> tensor<2xi32>
  return %crs : tensor<2xi32>
// CHECK: return %[[MUL]] : tensor<2xi32>
}

// -----

// CHECK-LABEL: func @compute_reshape_shape
func @compute_reshape_shape(%arg0: tensor<2xi32>, %arg1: index) -> tensor<2xi32> {
  %mul = mhlo.multiply %arg0, %arg0 : tensor<2xi32>
  %crs = mhlo.compute_reshape_shape %arg1, %mul : index, tensor<2xi32> -> tensor<2xi32>
// CHECK: mhlo.compute_reshape_shape
  return %crs : tensor<2xi32>
}

// -----

// CHECK-LABEL: @redundant_cstr_reshapable
func @redundant_cstr_reshapable(%arg0 : tensor<?x8x?x64xf32>)
    -> !shape.witness {
  // CHECK: %[[WITNESS:.*]] = shape.const_witness true
  // CHECK: return %[[WITNESS]] : !shape.witness
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c512 = arith.constant 512 : index
  %s0 = shape.shape_of %arg0 : tensor<?x8x?x64xf32> -> tensor<4xindex>
  %n0 = shape.num_elements %s0 : tensor<4xindex> -> index
  %dim00 = tensor.dim %arg0, %c0 : tensor<?x8x?x64xf32>
  %dim02 = tensor.dim %arg0, %c2 : tensor<?x8x?x64xf32>
  %s1_ = tensor.from_elements %dim02, %dim00, %c512 : tensor<3xindex>
  %s1 = arith.index_cast %s1_ : tensor<3xindex> to tensor<3xi32>
  %w = mhlo.cstr_reshapable %n0, %s1 : index, tensor<3xi32>
  return %w : !shape.witness
}

// -----

// CHECK-LABEL: @redundant_cstr_reshapable_less_obvious
func @redundant_cstr_reshapable_less_obvious(%arg0 : tensor<?x4x?x64xf32>)
    -> !shape.witness {
  // CHECK: %[[WITNESS:.*]] = shape.const_witness true
  // CHECK: return %[[WITNESS]] : !shape.witness
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c128 = arith.constant 128 : i32
  %s0 = shape.shape_of %arg0 : tensor<?x4x?x64xf32> -> tensor<4xindex>
  %n0 = shape.num_elements %s0 : tensor<4xindex> -> index
  %dim00 = tensor.dim %arg0, %c0 : tensor<?x4x?x64xf32>
  %dim00_twice = arith.muli %c2, %dim00 : index
  %dim00_twice_ = arith.index_cast %dim00_twice : index to i32
  %dim02 = tensor.dim %arg0, %c2 : tensor<?x4x?x64xf32>
  %dim02_ = arith.index_cast %dim02 : index to i32
  %s1 = tensor.from_elements %dim02_, %dim00_twice_, %c128 : tensor<3xi32>
  %w = mhlo.cstr_reshapable %n0, %s1 : index, tensor<3xi32>
  return %w : !shape.witness
}

// -----

// CHECK-LABEL: @redundant_cstr_reshapable
func @redundant_cstr_reshapable(%arg0 : tensor<?x8x?x64xf32>)
    -> !shape.witness {
  // CHECK: %[[WITNESS:.*]] = shape.const_witness true
  // CHECK: return %[[WITNESS]] : !shape.witness
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c128 = arith.constant 128 : i32
  %cminus1 = arith.constant -1 : i32
  %s0 = shape.shape_of %arg0 : tensor<?x8x?x64xf32> -> tensor<4xindex>
  %n0 = shape.num_elements %s0 : tensor<4xindex> -> index
  %dim02 = tensor.dim %arg0, %c2 : tensor<?x8x?x64xf32>
  %dim02_ = arith.index_cast %dim02 : index to i32
  %s1 = tensor.from_elements %dim02_, %cminus1, %c128 : tensor<3xi32>
  %w = mhlo.cstr_reshapable %n0, %s1 : index, tensor<3xi32>
  return %w : !shape.witness
}

// -----

// CHECK-LABEL: @nonredundant_cstr_reshapable
func @nonredundant_cstr_reshapable(%arg0 : tensor<?x8x?x64xf32>)
    -> !shape.witness {
  // CHECK: %[[WITNESS:.*]] = mhlo.cstr_reshapable %{{.*}}, %{{.*}}
  // CHECK: return %[[WITNESS]] : !shape.witness
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c42 = arith.constant 42 : i32
  %cminus1 = arith.constant -1 : i32
  %s0 = shape.shape_of %arg0 : tensor<?x8x?x64xf32> -> tensor<4xindex>
  %n0 = shape.num_elements %s0 : tensor<4xindex> -> index
  %dim02 = tensor.dim %arg0, %c2 : tensor<?x8x?x64xf32>
  %dim02_ = arith.index_cast %dim02 : index to i32
  %s1 = tensor.from_elements %dim02_, %cminus1, %c42 : tensor<3xi32>
  %w = mhlo.cstr_reshapable %n0, %s1 : index, tensor<3xi32>
  return %w : !shape.witness
}

// -----

// CHECK-LABEL: @redundant_cstr_reshapable
func @redundant_cstr_reshapable(%arg0 : tensor<?x8x?x64xf32>)
    -> !shape.witness {
  // CHECK: %[[WITNESS:.*]] = shape.const_witness true
  // CHECK: return %[[WITNESS]] : !shape.witness
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : i32
  %cminus1 = arith.constant -1 : i32
  %s0 = shape.shape_of %arg0 : tensor<?x8x?x64xf32> -> tensor<4xindex>
  %n0 = shape.num_elements %s0 : tensor<4xindex> -> index
  %dim00 = tensor.dim %arg0, %c0 : tensor<?x8x?x64xf32>
  %dim02 = tensor.dim %arg0, %c2 : tensor<?x8x?x64xf32>
  %dim00_ = arith.index_cast %dim00 : index to i32
  %dim02_ = arith.index_cast %dim02 : index to i32
  %s1 = tensor.from_elements %dim02_, %dim00_ , %c64, %cminus1 : tensor<4xi32>
  %w = mhlo.cstr_reshapable %n0, %s1 : index, tensor<4xi32>
  return %w : !shape.witness
}

// -----

// CHECK-LABEL: @dynamic_reshape_to_collapse_shape
// CHECK-SAME: %[[ARG:.*]]: tensor<1x4x?x64x?x8x1x1xf32>
func @dynamic_reshape_to_collapse_shape(%arg0 : tensor<1x4x?x64x?x8x1x1xf32>)
    -> tensor<?x?x8xf32> {
  // CHECK: %[[RESULT:.*]] = tensor.collapse_shape %[[ARG]] {{\[}}[0, 1, 2], [3, 4], [5, 6, 7]{{\]}}
  // CHECK: return %[[RESULT]]
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  %c8_i32 = arith.constant 8 : i32
  %c64_i32 = arith.constant 64 : i32
  %d2 = tensor.dim %arg0, %c2 : tensor<1x4x?x64x?x8x1x1xf32>
  %d4 = tensor.dim %arg0, %c4 : tensor<1x4x?x64x?x8x1x1xf32>
  %d2_i32 = arith.index_cast %d2 : index to i32
  %d4_i32 = arith.index_cast %d4 : index to i32
  %s0 = arith.muli %c4_i32, %d2_i32 : i32
  %s1 = arith.muli %c64_i32, %d4_i32 : i32
  %shape = tensor.from_elements %s0, %s1, %c8_i32 : tensor<3xi32>
  %result = "mhlo.dynamic_reshape"(%arg0, %shape)
      : (tensor<1x4x?x64x?x8x1x1xf32>, tensor<3xi32>) -> tensor<?x?x8xf32>
  return %result : tensor<?x?x8xf32>
}

// -----

// CHECK-LABEL: func @reshape_integration(
// CHECK-SAME:      %arg0: tensor<512x512xf32>,
// CHECK-SAME:      %arg1: tensor<?x8x?x64xf32>,
// CHECK-SAME:      %[[DYN_SHAPE:.*]]: tensor<4xi32>,
// CHECK-SAME:      %arg3: tensor<512xf32>,
// CHECK-SAME:      %arg4: tensor<?x?x512xf32>,
// CHECK-SAME:      %arg5: tensor<512xf32>,
// CHECK-SAME:      %arg6: tensor<512xf32>,
// CHECK-SAME:      %arg7: tensor<512x2048xf32>,
// CHECK-SAME:      %arg8: tensor<2048xf32>,
// CHECK-SAME:      %arg9: tensor<2048x512xf32>,
// CHECK-SAME:      %arg10: tensor<512xf32>,
// CHECK-SAME:      %arg11: tensor<512xf32>,
// CHECK-SAME:      %arg12: tensor<512xf32>)
func @reshape_integration(%arg0: tensor<512x512xf32>, %arg1: tensor<?x8x?x64xf32>, %arg2: tensor<4xi32>, %arg3: tensor<512xf32>, %arg4: tensor<?x?x512xf32>, %arg5: tensor<512xf32>, %arg6: tensor<512xf32>, %arg7: tensor<512x2048xf32>, %arg8: tensor<2048xf32>, %arg9: tensor<2048x512xf32>, %arg10: tensor<512xf32>, %arg11: tensor<512xf32>, %arg12: tensor<512xf32>) -> tensor<?x512xf32> {
  %0 = mhlo.constant dense<512> : tensor<1xi32>
  %1 = shape.shape_of %arg1 : tensor<?x8x?x64xf32> -> tensor<4xindex>
  %2 = shape.num_elements %1 : tensor<4xindex> -> index
  // CHECK: %[[W:.*]] = mhlo.cstr_reshapable
  %3 = mhlo.cstr_reshapable %2, %arg2 : index, tensor<4xi32>
  // CHECK: shape.assuming %[[W]]
  %4 = shape.assuming %3 -> (tensor<?x8x?x64xf32>) {
    // CHECK: %[[SHAPE:.*]] = mhlo.compute_reshape_shape %{{.*}}, %[[DYN_SHAPE]]
    %20 = mhlo.compute_reshape_shape %2, %arg2 : index, tensor<4xi32> -> tensor<4xi32>
    // CHECK: "mhlo.dynamic_reshape"(%arg1, %[[SHAPE]])
    %21 = "mhlo.dynamic_reshape"(%arg1, %20) : (tensor<?x8x?x64xf32>, tensor<4xi32>) -> tensor<?x8x?x64xf32>
    // CHECK: shape.assuming_yield
    shape.assuming_yield %21 : tensor<?x8x?x64xf32>
  }
  %5 = "mhlo.transpose"(%4) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<?x8x?x64xf32>) -> tensor<?x?x8x64xf32>
  %6 = "mhlo.transpose"(%5) {permutation = dense<[0, 1, 3, 2]> : tensor<4xi64>} : (tensor<?x?x8x64xf32>) -> tensor<?x?x64x8xf32>
  %7 = shape.shape_of %6 : tensor<?x?x64x8xf32> -> tensor<4xindex>
  %8 = arith.index_cast %7 : tensor<4xindex> to tensor<4xi32>
  %9 = "mhlo.slice"(%8) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xi32>) -> tensor<1xi32>
  %10 = "mhlo.reshape"(%9) : (tensor<1xi32>) -> tensor<i32>
  %11 = "mhlo.slice"(%8) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<4xi32>) -> tensor<1xi32>
  %12 = "mhlo.reshape"(%11) : (tensor<1xi32>) -> tensor<i32>
  %13 = mhlo.multiply %10, %12 : tensor<i32>
  %14 = "mhlo.reshape"(%13) : (tensor<i32>) -> tensor<1xi32>
  %15 = "mhlo.concatenate"(%14, %0) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %16 = shape.shape_of %6 : tensor<?x?x64x8xf32> -> tensor<4xindex>
  %17 = shape.num_elements %16 : tensor<4xindex> -> index
  // CHECK-NOT: cstr_reshapable
  %18 = mhlo.cstr_reshapable %17, %15 : index, tensor<2xi32>
  // CHECK-NOT: assuming
  %19 = shape.assuming %18 -> (tensor<?x512xf32>) {
    // CHECK-NOT: compute_reshape_shape
    %20 = mhlo.compute_reshape_shape %17, %15 : index, tensor<2xi32> -> tensor<2xi32>
    // CHECK: tensor.collapse_shape
    %21 = "mhlo.dynamic_reshape"(%6, %20) : (tensor<?x?x64x8xf32>, tensor<2xi32>) -> tensor<?x512xf32>
    // CHECK-NOT: assuming_yield
    shape.assuming_yield %21 : tensor<?x512xf32>
  }
  return %19 : tensor<?x512xf32>
}
