// RUN: xla-opt %s --legalize-stablehlo-to-tensor-ir --split-input-file | FileCheck %s

// CHECK-LABEL: nv_tensor_ir.graph @reshape
// CHECK-SAME: (%[[IN:.*]]: tensor<2x3xf32>) -> tensor<6x1xf32>
// CHECK-NEXT: %[[RES:.*]] = reshape %[[IN]] : tensor<2x3xf32> -> tensor<6x1xf32>
// CHECK-NEXT: results %[[RES]] : tensor<6x1xf32>
func.func @reshape(%arg0: tensor<2x3xf32>) -> tensor<6x1xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<2x3xf32>) -> tensor<6x1xf32>
  return %0 : tensor<6x1xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @reshape_i32
// CHECK-SAME: (%[[IN:.*]]: tensor<2x3xsi32>) -> tensor<3x2xsi32>
// CHECK-NEXT: %[[RES:.*]] = reshape %[[IN]] : tensor<2x3xsi32> -> tensor<3x2xsi32>
// CHECK-NEXT: results %[[RES]] : tensor<3x2xsi32>
func.func @reshape_i32(%arg0: tensor<2x3xi32>) -> tensor<3x2xi32> {
  %0 = stablehlo.reshape %arg0 : (tensor<2x3xi32>) -> tensor<3x2xi32>
  return %0 : tensor<3x2xi32>
}

// -----

// The permutation is forwarded as is. With the opposite convention the result
// shape would be 3x4x2 and the nv_tensor_ir.transpose verifier would reject it.
// CHECK-LABEL: nv_tensor_ir.graph @transpose
// CHECK-SAME: (%[[IN:.*]]: tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
// CHECK-NEXT: %[[RES:.*]] = transpose %[[IN]] permutation = [2, 0, 1] : tensor<2x3x4xf32> -> tensor<4x2x3xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4x2x3xf32>
func.func @transpose(%arg0: tensor<2x3x4xf32>) -> tensor<4x2x3xf32> {
  %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
  return %0 : tensor<4x2x3xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @slice
// CHECK-SAME: (%[[IN:.*]]: tensor<8x8xf32>) -> tensor<4x2xf32>
// CHECK-NEXT: %[[RES:.*]] = slice %[[IN]] starts = [0, 1] limits = [8, 7] strides = [2, 3] : tensor<8x8xf32> -> tensor<4x2xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4x2xf32>
func.func @slice(%arg0: tensor<8x8xf32>) -> tensor<4x2xf32> {
  %0 = stablehlo.slice %arg0 [0:8:2, 1:7:3] : (tensor<8x8xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @concatenate
// CHECK-SAME: (%[[LHS:.*]]: tensor<2x2xf32>, %[[RHS:.*]]: tensor<2x3xf32>) -> tensor<2x5xf32>
// CHECK-NEXT: %[[RES:.*]] = concatenate %[[LHS]], %[[RHS]] dimension = 1 : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<2x5xf32>
// CHECK-NEXT: results %[[RES]] : tensor<2x5xf32>
func.func @concatenate(%arg0: tensor<2x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x5xf32> {
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<2x5xf32>
  return %0 : tensor<2x5xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @iota
// CHECK-SAME: () -> tensor<2x3xf32>
// CHECK-NEXT: %[[RES:.*]] = iota dimension = 1 : tensor<2x3xf32>
// CHECK-NEXT: results %[[RES]] : tensor<2x3xf32>
func.func @iota() -> tensor<2x3xf32> {
  %0 = stablehlo.iota dim = 1 : tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @iota_i32
// CHECK-SAME: () -> tensor<4x5xsi32>
// CHECK-NEXT: %[[RES:.*]] = iota dimension = 0 : tensor<4x5xsi32>
// CHECK-NEXT: results %[[RES]] : tensor<4x5xsi32>
func.func @iota_i32() -> tensor<4x5xi32> {
  %0 = stablehlo.iota dim = 0 : tensor<4x5xi32>
  return %0 : tensor<4x5xi32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @constant_f32
// CHECK-SAME: () -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = constant dense<1.000000e+00> : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @constant_f32() -> tensor<4xf32> {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// The value attribute is rebuilt over the converted element type, because
// nv_tensor_ir.constant infers its result type from the attribute.
// CHECK-LABEL: nv_tensor_ir.graph @constant_i32
// CHECK-SAME: () -> tensor<4xsi32>
// CHECK-NEXT: %[[RES:.*]] = constant dense<[1, 2, 3, 4]> : tensor<4xsi32>
// CHECK-NEXT: results %[[RES]] : tensor<4xsi32>
func.func @constant_i32() -> tensor<4xi32> {
  %0 = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @constant_i32_splat
// CHECK-SAME: () -> tensor<2x2xsi32>
// CHECK-NEXT: %[[RES:.*]] = constant dense<7> : tensor<2x2xsi32>
// CHECK-NEXT: results %[[RES]] : tensor<2x2xsi32>
func.func @constant_i32_splat() -> tensor<2x2xi32> {
  %0 = stablehlo.constant dense<7> : tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @constant_i64
// CHECK-SAME: () -> tensor<2xsi64>
// CHECK-NEXT: %[[RES:.*]] = constant dense<[-1, 9223372036854775807]> : tensor<2xsi64>
// CHECK-NEXT: results %[[RES]] : tensor<2xsi64>
func.func @constant_i64() -> tensor<2xi64> {
  %0 = stablehlo.constant dense<[-1, 9223372036854775807]> : tensor<2xi64>
  return %0 : tensor<2xi64>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @constant_ui32
// CHECK-SAME: () -> tensor<2xui32>
// CHECK-NEXT: %[[RES:.*]] = constant dense<[1, 4294967295]> : tensor<2xui32>
// CHECK-NEXT: results %[[RES]] : tensor<2xui32>
func.func @constant_ui32() -> tensor<2xui32> {
  %0 = stablehlo.constant dense<[1, 4294967295]> : tensor<2xui32>
  return %0 : tensor<2xui32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @constant_i1
// CHECK-SAME: () -> tensor<2xi1>
// CHECK-NEXT: %[[RES:.*]] = constant dense<[true, false]> : tensor<2xi1>
// CHECK-NEXT: results %[[RES]] : tensor<2xi1>
func.func @constant_i1() -> tensor<2xi1> {
  %0 = stablehlo.constant dense<[true, false]> : tensor<2xi1>
  return %0 : tensor<2xi1>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @broadcast_leading_dim
// CHECK-SAME: (%[[IN:.*]]: tensor<3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT: %[[EXPANDED:.*]] = reshape %[[IN]] : tensor<3xf32> -> tensor<1x3xf32>
// CHECK-NEXT: %[[RES:.*]] = broadcast %[[EXPANDED]] : tensor<1x3xf32> -> tensor<2x3xf32>
// CHECK-NEXT: results %[[RES]] : tensor<2x3xf32>
func.func @broadcast_leading_dim(%arg0: tensor<3xf32>) -> tensor<2x3xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @broadcast_trailing_dim
// CHECK-SAME: (%[[IN:.*]]: tensor<3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT: %[[EXPANDED:.*]] = reshape %[[IN]] : tensor<3xf32> -> tensor<3x1xf32>
// CHECK-NEXT: %[[RES:.*]] = broadcast %[[EXPANDED]] : tensor<3x1xf32> -> tensor<3x2xf32>
// CHECK-NEXT: results %[[RES]] : tensor<3x2xf32>
func.func @broadcast_trailing_dim(%arg0: tensor<3xf32>) -> tensor<3x2xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<3xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// -----

// The input already has the output rank, so no reshape is needed.
// CHECK-LABEL: nv_tensor_ir.graph @broadcast_unit_dim
// CHECK-SAME: (%[[IN:.*]]: tensor<16x1x64xf32>) -> tensor<16x32x64xf32>
// CHECK-NEXT: %[[RES:.*]] = broadcast %[[IN]] : tensor<16x1x64xf32> -> tensor<16x32x64xf32>
// CHECK-NEXT: results %[[RES]] : tensor<16x32x64xf32>
func.func @broadcast_unit_dim(%arg0: tensor<16x1x64xf32>) -> tensor<16x32x64xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<16x1x64xf32>) -> tensor<16x32x64xf32>
  return %0 : tensor<16x32x64xf32>
}

// -----

// A no-op broadcast is rejected by the nv_tensor_ir.broadcast verifier, so no
// operation is emitted at all.
// CHECK-LABEL: nv_tensor_ir.graph @broadcast_noop
// CHECK-SAME: (%[[IN:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: results %[[IN]] : tensor<4xf32>
func.func @broadcast_noop(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// A rank-0 argument is declared as `[1]` on the boundary, so the reshape that
// expands it for the broadcast starts from rank 1. See `PromoteRank0Tensors`.
// CHECK-LABEL: nv_tensor_ir.graph @broadcast_scalar
// CHECK-SAME: (%[[IN:.*]]: tensor<1xf32>) -> tensor<2x3xf32>
// CHECK-NEXT: %[[EXPANDED:.*]] = reshape %[[IN]] : tensor<1xf32> -> tensor<1x1xf32>
// CHECK-NEXT: %[[RES:.*]] = broadcast %[[EXPANDED]] : tensor<1x1xf32> -> tensor<2x3xf32>
// CHECK-NEXT: results %[[RES]] : tensor<2x3xf32>
func.func @broadcast_scalar(%arg0: tensor<f32>) -> tensor<2x3xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// -----

// The broadcast dimensions are not ascending, so the input dimensions have to
// be sorted with a transpose before the reshape.
// CHECK-LABEL: nv_tensor_ir.graph @broadcast_permuted
// CHECK-SAME: (%[[IN:.*]]: tensor<2x3xf32>) -> tensor<4x3x2xf32>
// CHECK-NEXT: %[[SORTED:.*]] = transpose %[[IN]] permutation = [1, 0] : tensor<2x3xf32> -> tensor<3x2xf32>
// CHECK-NEXT: %[[EXPANDED:.*]] = reshape %[[SORTED]] : tensor<3x2xf32> -> tensor<1x3x2xf32>
// CHECK-NEXT: %[[RES:.*]] = broadcast %[[EXPANDED]] : tensor<1x3x2xf32> -> tensor<4x3x2xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4x3x2xf32>
func.func @broadcast_permuted(%arg0: tensor<2x3xf32>) -> tensor<4x3x2xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [2, 1] : (tensor<2x3xf32>) -> tensor<4x3x2xf32>
  return %0 : tensor<4x3x2xf32>
}
