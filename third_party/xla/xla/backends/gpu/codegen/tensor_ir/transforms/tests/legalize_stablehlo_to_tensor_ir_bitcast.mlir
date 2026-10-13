// RUN: xla-opt %s --legalize-stablehlo-to-tensor-ir --split-input-file | FileCheck %s

// mhlo.bitcast reinterprets a buffer under a new shape and layout without
// moving any element. nv_tensor_ir tensors are always row-major, so the
// lowering normalizes both sides to row-major and relinearizes in between:
// transpose into physical order, reshape, transpose back into logical order.
// Steps that would be the identity are skipped.
//
// The layouts come from the `source_layout` and `result_layout` attributes the
// HLO importer attaches to every mhlo.bitcast. They are discardable attributes
// (mhlo.bitcast declares none in ODS), so they are the only record of the
// physical element order in the imported module.

// Both sides carry a non-default layout, and the two physical shapes agree, so
// the relinearizing reshape is skipped and only the two transposes remain.
// CHECK-LABEL: nv_tensor_ir.graph @bitcast_both_non_default
// CHECK-SAME: (%[[IN:.*]]: tensor<8x16x32xf32>) -> tensor<16x8x32xf32>
// CHECK-NEXT: %[[PHYSICAL:.*]] = transpose %[[IN]] permutation = [2, 1, 0] : tensor<8x16x32xf32> -> tensor<32x16x8xf32>
// CHECK-NEXT: %[[RES:.*]] = transpose %[[PHYSICAL]] permutation = [1, 2, 0] : tensor<32x16x8xf32> -> tensor<16x8x32xf32>
// CHECK-NEXT: results %[[RES]] : tensor<16x8x32xf32>
func.func @bitcast_both_non_default(%arg0: tensor<8x16x32xf32>) -> tensor<16x8x32xf32> {
  %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0, 2]> : tensor<3xindex>, source_layout = dense<[0, 1, 2]> : tensor<3xindex>} : (tensor<8x16x32xf32>) -> tensor<16x8x32xf32>
  return %0 : tensor<16x8x32xf32>
}

// -----

// Both sides are row-major already, so the bitcast is a plain reshape.
// CHECK-LABEL: nv_tensor_ir.graph @bitcast_default_layouts
// CHECK-SAME: (%[[IN:.*]]: tensor<8x16xf32>) -> tensor<8x4x4xf32>
// CHECK-NEXT: %[[RES:.*]] = reshape %[[IN]] : tensor<8x16xf32> -> tensor<8x4x4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<8x4x4xf32>
func.func @bitcast_default_layouts(%arg0: tensor<8x16xf32>) -> tensor<8x4x4xf32> {
  %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<8x16xf32>) -> tensor<8x4x4xf32>
  return %0 : tensor<8x4x4xf32>
}

// -----

// Only the operand is stored transposed, so only the first transpose is
// emitted and the trailing one is skipped.
// CHECK-LABEL: nv_tensor_ir.graph @bitcast_source_non_default
// CHECK-SAME: (%[[IN:.*]]: tensor<8x16xf32>) -> tensor<4x4x8xf32>
// CHECK-NEXT: %[[PHYSICAL:.*]] = transpose %[[IN]] permutation = [1, 0] : tensor<8x16xf32> -> tensor<16x8xf32>
// CHECK-NEXT: %[[RES:.*]] = reshape %[[PHYSICAL]] : tensor<16x8xf32> -> tensor<4x4x8xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4x4x8xf32>
func.func @bitcast_source_non_default(%arg0: tensor<8x16xf32>) -> tensor<4x4x8xf32> {
  %0 = mhlo.bitcast %arg0 {result_layout = dense<[2, 1, 0]> : tensor<3xindex>, source_layout = dense<[0, 1]> : tensor<2xindex>} : (tensor<8x16xf32>) -> tensor<4x4x8xf32>
  return %0 : tensor<4x4x8xf32>
}

// -----

// Only the result is stored transposed.
// CHECK-LABEL: nv_tensor_ir.graph @bitcast_result_non_default
// CHECK-SAME: (%[[IN:.*]]: tensor<8x16xf32>) -> tensor<16x8xf32>
// CHECK-NEXT: %[[RES:.*]] = transpose %[[IN]] permutation = [1, 0] : tensor<8x16xf32> -> tensor<16x8xf32>
// CHECK-NEXT: results %[[RES]] : tensor<16x8xf32>
func.func @bitcast_result_non_default(%arg0: tensor<8x16xf32>) -> tensor<16x8xf32> {
  %0 = mhlo.bitcast %arg0 {result_layout = dense<[0, 1]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<8x16xf32>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

// Same shape, same default layout: every step is the identity, so the bitcast
// disappears entirely.
// CHECK-LABEL: nv_tensor_ir.graph @bitcast_identity
// CHECK-SAME: (%[[IN:.*]]: tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: results %[[IN]] : tensor<8x16xf32>
func.func @bitcast_identity(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = mhlo.bitcast %arg0 {result_layout = dense<[1, 0]> : tensor<2xindex>, source_layout = dense<[1, 0]> : tensor<2xindex>} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
