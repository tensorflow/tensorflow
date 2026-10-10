// RUN: xla-opt %s --legalize-stablehlo-to-tensor-ir --split-input-file | FileCheck %s

// CHECK-LABEL: nv_tensor_ir.graph @matmul
// CHECK-SAME: (%[[LHS:.*]]: tensor<8x16xf32>, %[[RHS:.*]]: tensor<16x32xf32>) -> tensor<8x32xf32>
// CHECK-NEXT: %[[RES:.*]] = matmul(%[[LHS]], %[[RHS]]) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
// CHECK-NEXT: results %[[RES]] : tensor<8x32xf32>
func.func @matmul(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<8x32xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @matmul_i32
// CHECK-SAME: (%[[LHS:.*]]: tensor<8x16xsi32>, %[[RHS:.*]]: tensor<16x32xsi32>) -> tensor<8x32xsi32>
// CHECK-NEXT: %[[RES:.*]] = matmul(%[[LHS]], %[[RHS]]) : (tensor<8x16xsi32>, tensor<16x32xsi32>) -> tensor<8x32xsi32>
// CHECK-NEXT: results %[[RES]] : tensor<8x32xsi32>
func.func @matmul_i32(%arg0: tensor<8x16xi32>, %arg1: tensor<16x32xi32>) -> tensor<8x32xi32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<8x16xi32>, tensor<16x32xi32>) -> tensor<8x32xi32>
  return %0 : tensor<8x32xi32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @matmul_batched
// CHECK-SAME: (%[[LHS:.*]]: tensor<8x16x32xf32>, %[[RHS:.*]]: tensor<8x32x48xf32>) -> tensor<8x16x48xf32>
// CHECK-NEXT: %[[RES:.*]] = matmul(%[[LHS]], %[[RHS]]) : (tensor<8x16x32xf32>, tensor<8x32x48xf32>) -> tensor<8x16x48xf32>
// CHECK-NEXT: results %[[RES]] : tensor<8x16x48xf32>
func.func @matmul_batched(%arg0: tensor<8x16x32xf32>, %arg1: tensor<8x32x48xf32>) -> tensor<8x16x48xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x16x32xf32>, tensor<8x32x48xf32>) -> tensor<8x16x48xf32>
  return %0 : tensor<8x16x48xf32>
}

// -----

// The lhs contracts along its leading dimension, so it is transposed into the
// (M, K) layout that nv_tensor_ir.matmul requires.
// CHECK-LABEL: nv_tensor_ir.graph @matmul_lhs_transposed
// CHECK-SAME: (%[[LHS:.*]]: tensor<16x8xf32>, %[[RHS:.*]]: tensor<16x32xf32>) -> tensor<8x32xf32>
// CHECK-NEXT: %[[LHS_T:.*]] = transpose %[[LHS]] permutation = [1, 0] : tensor<16x8xf32> -> tensor<8x16xf32>
// CHECK-NEXT: %[[RES:.*]] = matmul(%[[LHS_T]], %[[RHS]]) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
// CHECK-NEXT: results %[[RES]] : tensor<8x32xf32>
func.func @matmul_lhs_transposed(%arg0: tensor<16x8xf32>, %arg1: tensor<16x32xf32>) -> tensor<8x32xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<16x8xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @matmul_rhs_transposed
// CHECK-SAME: (%[[LHS:.*]]: tensor<8x16xf32>, %[[RHS:.*]]: tensor<32x16xf32>) -> tensor<8x32xf32>
// CHECK-NEXT: %[[RHS_T:.*]] = transpose %[[RHS]] permutation = [1, 0] : tensor<32x16xf32> -> tensor<16x32xf32>
// CHECK-NEXT: %[[RES:.*]] = matmul(%[[LHS]], %[[RHS_T]]) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
// CHECK-NEXT: results %[[RES]] : tensor<8x32xf32>
func.func @matmul_rhs_transposed(%arg0: tensor<8x16xf32>, %arg1: tensor<32x16xf32>) -> tensor<8x32xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1] : (tensor<8x16xf32>, tensor<32x16xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @matmul_batched_both_transposed
// CHECK-SAME: (%[[LHS:.*]]: tensor<4x32x16xf32>, %[[RHS:.*]]: tensor<4x48x32xf32>) -> tensor<4x16x48xf32>
// CHECK-NEXT: %[[LHS_T:.*]] = transpose %[[LHS]] permutation = [0, 2, 1] : tensor<4x32x16xf32> -> tensor<4x16x32xf32>
// CHECK-NEXT: %[[RHS_T:.*]] = transpose %[[RHS]] permutation = [0, 2, 1] : tensor<4x48x32xf32> -> tensor<4x32x48xf32>
// CHECK-NEXT: %[[RES:.*]] = matmul(%[[LHS_T]], %[[RHS_T]]) : (tensor<4x16x32xf32>, tensor<4x32x48xf32>) -> tensor<4x16x48xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4x16x48xf32>
func.func @matmul_batched_both_transposed(%arg0: tensor<4x32x16xf32>, %arg1: tensor<4x48x32xf32>) -> tensor<4x16x48xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [1] x [2] : (tensor<4x32x16xf32>, tensor<4x48x32xf32>) -> tensor<4x16x48xf32>
  return %0 : tensor<4x16x48xf32>
}

// -----

//===----------------------------------------------------------------------===//
// The nine cases of the old string-building converter's specification,
// .../codegen/tensor_ir/tests/dot.hlo, ported to StableHLO. The op sequence
// below must keep matching what that file pins.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: nv_tensor_ir.graph @test_dot_no_batch
// CHECK-SAME: (%[[LHS:.*]]: tensor<8x32xf32>, %[[RHS:.*]]: tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: %[[RES:.*]] = matmul(%[[LHS]], %[[RHS]]) : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: results %[[RES]] : tensor<8x16xf32>
func.func @test_dot_no_batch(%arg0: tensor<8x32xf32>, %arg1: tensor<32x16xf32>) -> tensor<8x16xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @test_dot_with_batch
// CHECK-SAME: (%[[LHS:.*]]: tensor<4x8x32xf32>, %[[RHS:.*]]: tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
// CHECK-NEXT: %[[RES:.*]] = matmul(%[[LHS]], %[[RHS]]) : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4x8x16xf32>
func.func @test_dot_with_batch(%arg0: tensor<4x8x32xf32>, %arg1: tensor<4x32x16xf32>) -> tensor<4x8x16xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// -----

// A batch dimension that is not leading.
// CHECK-LABEL: nv_tensor_ir.graph @test_dot_transpose_lhs
// CHECK-SAME: (%[[LHS:.*]]: tensor<32x4x8xf32>, %[[RHS:.*]]: tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
// CHECK-NEXT: %[[LHS_T:.*]] = transpose %[[LHS]] permutation = [1, 2, 0] : tensor<32x4x8xf32> -> tensor<4x8x32xf32>
// CHECK-NEXT: %[[RES:.*]] = matmul(%[[LHS_T]], %[[RHS]]) : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4x8x16xf32>
func.func @test_dot_transpose_lhs(%arg0: tensor<32x4x8xf32>, %arg1: tensor<4x32x16xf32>) -> tensor<4x8x16xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [1] x [0], contracting_dims = [0] x [1] : (tensor<32x4x8xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @test_dot_transpose_rhs
// CHECK-SAME: (%[[LHS:.*]]: tensor<4x8x32xf32>, %[[RHS:.*]]: tensor<32x4x16xf32>) -> tensor<4x8x16xf32>
// CHECK-NEXT: %[[RHS_T:.*]] = transpose %[[RHS]] permutation = [1, 0, 2] : tensor<32x4x16xf32> -> tensor<4x32x16xf32>
// CHECK-NEXT: %[[RES:.*]] = matmul(%[[LHS]], %[[RHS_T]]) : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4x8x16xf32>
func.func @test_dot_transpose_rhs(%arg0: tensor<4x8x32xf32>, %arg1: tensor<32x4x16xf32>) -> tensor<4x8x16xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [1], contracting_dims = [2] x [0] : (tensor<4x8x32xf32>, tensor<32x4x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// -----

// Multiple batch dimensions, collapsed into one.
// CHECK-LABEL: nv_tensor_ir.graph @test_dot_reshape_batch
// CHECK-SAME: (%[[LHS:.*]]: tensor<2x4x8x32xf32>, %[[RHS:.*]]: tensor<2x4x32x16xf32>) -> tensor<2x4x8x16xf32>
// CHECK-NEXT: %[[LHS_R:.*]] = reshape %[[LHS]] : tensor<2x4x8x32xf32> -> tensor<8x8x32xf32>
// CHECK-NEXT: %[[RHS_R:.*]] = reshape %[[RHS]] : tensor<2x4x32x16xf32> -> tensor<8x32x16xf32>
// CHECK-NEXT: %[[MATMUL:.*]] = matmul(%[[LHS_R]], %[[RHS_R]]) : (tensor<8x8x32xf32>, tensor<8x32x16xf32>) -> tensor<8x8x16xf32>
// CHECK-NEXT: %[[RES:.*]] = reshape %[[MATMUL]] : tensor<8x8x16xf32> -> tensor<2x4x8x16xf32>
// CHECK-NEXT: results %[[RES]] : tensor<2x4x8x16xf32>
func.func @test_dot_reshape_batch(%arg0: tensor<2x4x8x32xf32>, %arg1: tensor<2x4x32x16xf32>) -> tensor<2x4x8x16xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<2x4x8x32xf32>, tensor<2x4x32x16xf32>) -> tensor<2x4x8x16xf32>
  return %0 : tensor<2x4x8x16xf32>
}

// -----

// Multiple contracting dimensions, merged into one.
// CHECK-LABEL: nv_tensor_ir.graph @test_dot_reshape_contracting
// CHECK-SAME: (%[[LHS:.*]]: tensor<4x8x32x2xf32>, %[[RHS:.*]]: tensor<4x32x2x16xf32>) -> tensor<4x8x16xf32>
// CHECK-NEXT: %[[LHS_R:.*]] = reshape %[[LHS]] : tensor<4x8x32x2xf32> -> tensor<4x8x64xf32>
// CHECK-NEXT: %[[RHS_R:.*]] = reshape %[[RHS]] : tensor<4x32x2x16xf32> -> tensor<4x64x16xf32>
// CHECK-NEXT: %[[RES:.*]] = matmul(%[[LHS_R]], %[[RHS_R]]) : (tensor<4x8x64xf32>, tensor<4x64x16xf32>) -> tensor<4x8x16xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4x8x16xf32>
func.func @test_dot_reshape_contracting(%arg0: tensor<4x8x32x2xf32>, %arg1: tensor<4x32x2x16xf32>) -> tensor<4x8x16xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2, 3] x [1, 2] : (tensor<4x8x32x2xf32>, tensor<4x32x2x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// -----

// Two free dimensions on the lhs, merged into one.
// CHECK-LABEL: nv_tensor_ir.graph @test_dot_reshape_lhs_noncontracting
// CHECK-SAME: (%[[LHS:.*]]: tensor<4x8x8x32xf32>, %[[RHS:.*]]: tensor<4x32x16xf32>) -> tensor<4x8x8x16xf32>
// CHECK-NEXT: %[[LHS_R:.*]] = reshape %[[LHS]] : tensor<4x8x8x32xf32> -> tensor<4x64x32xf32>
// CHECK-NEXT: %[[MATMUL:.*]] = matmul(%[[LHS_R]], %[[RHS]]) : (tensor<4x64x32xf32>, tensor<4x32x16xf32>) -> tensor<4x64x16xf32>
// CHECK-NEXT: %[[RES:.*]] = reshape %[[MATMUL]] : tensor<4x64x16xf32> -> tensor<4x8x8x16xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4x8x8x16xf32>
func.func @test_dot_reshape_lhs_noncontracting(%arg0: tensor<4x8x8x32xf32>, %arg1: tensor<4x32x16xf32>) -> tensor<4x8x8x16xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [3] x [1] : (tensor<4x8x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x8x16xf32>
  return %0 : tensor<4x8x8x16xf32>
}

// -----

// Two free dimensions on the rhs, merged into one.
// CHECK-LABEL: nv_tensor_ir.graph @test_dot_reshape_rhs_noncontracting
// CHECK-SAME: (%[[LHS:.*]]: tensor<4x8x32xf32>, %[[RHS:.*]]: tensor<4x32x16x16xf32>) -> tensor<4x8x16x16xf32>
// CHECK-NEXT: %[[RHS_R:.*]] = reshape %[[RHS]] : tensor<4x32x16x16xf32> -> tensor<4x32x256xf32>
// CHECK-NEXT: %[[MATMUL:.*]] = matmul(%[[LHS]], %[[RHS_R]]) : (tensor<4x8x32xf32>, tensor<4x32x256xf32>) -> tensor<4x8x256xf32>
// CHECK-NEXT: %[[RES:.*]] = reshape %[[MATMUL]] : tensor<4x8x256xf32> -> tensor<4x8x16x16xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4x8x16x16xf32>
func.func @test_dot_reshape_rhs_noncontracting(%arg0: tensor<4x8x32xf32>, %arg1: tensor<4x32x16x16xf32>) -> tensor<4x8x16x16xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x8x32xf32>, tensor<4x32x16x16xf32>) -> tensor<4x8x16x16xf32>
  return %0 : tensor<4x8x16x16xf32>
}

// -----

// Non-leading and reordered batch dimensions, reordered multi-dimensional
// contraction, and multiple free dimensions on both operands, all at once.
// CHECK-LABEL: nv_tensor_ir.graph @test_dot_transpose_reshape_all
// CHECK-SAME: (%[[LHS:.*]]: tensor<9x8x5x4x3x2xf32>, %[[RHS:.*]]: tensor<2x6x8x3x7x9xf32>) -> tensor<2x3x5x4x6x7xf32>
// CHECK-NEXT: %[[LHS_T:.*]] = transpose %[[LHS]] permutation = [5, 4, 2, 3, 1, 0] : tensor<9x8x5x4x3x2xf32> -> tensor<2x3x5x4x8x9xf32>
// CHECK-NEXT: %[[LHS_R:.*]] = reshape %[[LHS_T]] : tensor<2x3x5x4x8x9xf32> -> tensor<6x20x72xf32>
// CHECK-NEXT: %[[RHS_T:.*]] = transpose %[[RHS]] permutation = [0, 3, 2, 5, 1, 4] : tensor<2x6x8x3x7x9xf32> -> tensor<2x3x8x9x6x7xf32>
// CHECK-NEXT: %[[RHS_R:.*]] = reshape %[[RHS_T]] : tensor<2x3x8x9x6x7xf32> -> tensor<6x72x42xf32>
// CHECK-NEXT: %[[MATMUL:.*]] = matmul(%[[LHS_R]], %[[RHS_R]]) : (tensor<6x20x72xf32>, tensor<6x72x42xf32>) -> tensor<6x20x42xf32>
// CHECK-NEXT: %[[RES:.*]] = reshape %[[MATMUL]] : tensor<6x20x42xf32> -> tensor<2x3x5x4x6x7xf32>
// CHECK-NEXT: results %[[RES]] : tensor<2x3x5x4x6x7xf32>
func.func @test_dot_transpose_reshape_all(%arg0: tensor<9x8x5x4x3x2xf32>, %arg1: tensor<2x6x8x3x7x9xf32>) -> tensor<2x3x5x4x6x7xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [5, 4] x [0, 3], contracting_dims = [1, 0] x [2, 5] : (tensor<9x8x5x4x3x2xf32>, tensor<2x6x8x3x7x9xf32>) -> tensor<2x3x5x4x6x7xf32>
  return %0 : tensor<2x3x5x4x6x7xf32>
}

// -----

// Multiple contracting dimensions without any batch dimension: the collapsed
// form is the plain rank-2 matmul, not a size-1 batched one.
// CHECK-LABEL: nv_tensor_ir.graph @matmul_multiple_contracting_dims
// CHECK-SAME: (%[[LHS:.*]]: tensor<4x8x16xf32>, %[[RHS:.*]]: tensor<8x16x32xf32>) -> tensor<4x32xf32>
// CHECK-NEXT: %[[LHS_R:.*]] = reshape %[[LHS]] : tensor<4x8x16xf32> -> tensor<4x128xf32>
// CHECK-NEXT: %[[RHS_R:.*]] = reshape %[[RHS]] : tensor<8x16x32xf32> -> tensor<128x32xf32>
// CHECK-NEXT: %[[RES:.*]] = matmul(%[[LHS_R]], %[[RHS_R]]) : (tensor<4x128xf32>, tensor<128x32xf32>) -> tensor<4x32xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4x32xf32>
func.func @matmul_multiple_contracting_dims(%arg0: tensor<4x8x16xf32>, %arg1: tensor<8x16x32xf32>) -> tensor<4x32xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1, 2] x [0, 1] : (tensor<4x8x16xf32>, tensor<8x16x32xf32>) -> tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// -----

// A trailing batch dimension on both operands.
// CHECK-LABEL: nv_tensor_ir.graph @matmul_trailing_batch_dim
// CHECK-SAME: (%[[LHS:.*]]: tensor<4x8x16xf32>, %[[RHS:.*]]: tensor<16x8x32xf32>) -> tensor<8x4x32xf32>
// CHECK-NEXT: %[[LHS_T:.*]] = transpose %[[LHS]] permutation = [1, 0, 2] : tensor<4x8x16xf32> -> tensor<8x4x16xf32>
// CHECK-NEXT: %[[RHS_T:.*]] = transpose %[[RHS]] permutation = [1, 0, 2] : tensor<16x8x32xf32> -> tensor<8x16x32xf32>
// CHECK-NEXT: %[[RES:.*]] = matmul(%[[LHS_T]], %[[RHS_T]]) : (tensor<8x4x16xf32>, tensor<8x16x32xf32>) -> tensor<8x4x32xf32>
// CHECK-NEXT: results %[[RES]] : tensor<8x4x32xf32>
func.func @matmul_trailing_batch_dim(%arg0: tensor<4x8x16xf32>, %arg1: tensor<16x8x32xf32>) -> tensor<8x4x32xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [1] x [1], contracting_dims = [2] x [0] : (tensor<4x8x16xf32>, tensor<16x8x32xf32>) -> tensor<8x4x32xf32>
  return %0 : tensor<8x4x32xf32>
}

// -----

// An all-DEFAULT precision_config asks for nothing in particular, so it is
// accepted and dropped. Anything else is rejected, see the invalid tests.
// CHECK-LABEL: nv_tensor_ir.graph @matmul_default_precision
// CHECK-SAME: (%[[LHS:.*]]: tensor<8x16xf32>, %[[RHS:.*]]: tensor<16x32xf32>) -> tensor<8x32xf32>
// CHECK-NEXT: %[[RES:.*]] = matmul(%[[LHS]], %[[RHS]]) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
// CHECK-NEXT: results %[[RES]] : tensor<8x32xf32>
func.func @matmul_default_precision(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<8x32xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}
