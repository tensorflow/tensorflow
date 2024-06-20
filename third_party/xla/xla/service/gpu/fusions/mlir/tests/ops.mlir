// R-UN: mlir_fusions_opt %s --split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RU-N: mlir_fusions_opt %s --split-input-file | mlir_fusions_opt --split-input-file | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir_fusions_opt %s --split-input-file --mlir-print-op-generic | mlir_fusions_opt --split-input-file | FileCheck %s

func.func @shared_and_sync() -> (tensor<2xf32>, tensor<2xf32>) {
  %shared1 = xla_gpu.allocate_shared : tensor<2xf32>
  %shared2 = xla_gpu.allocate_shared : tensor<2xf32>
  %sync:2 = xla_gpu.sync_threads %shared1, %shared2
    : tensor<2xf32>, tensor<2xf32>
  return %sync#0, %sync#1 : tensor<2xf32>, tensor<2xf32>
}
// CHECK-LABEL: @shared_and_sync
// CHECK-NEXT: allocate_shared
// CHECK-NEXT: allocate_shared
// CHECK-NEXT: sync_threads
// CHECK-NEXT: return

// -----

func.func @atomic_rmw(%in: tensor<2x3xf32>, %i: index, %j: index)
    -> (tensor<2x3xf32>) {
  %ret = xla_gpu.atomic_rmw %in[%i, %j] : tensor<2x3xf32> {
    ^bb0(%current : f32):
      %c42 = arith.constant 42.0 : f32
      %add = arith.addf %current, %c42 : f32
      xla_gpu.yield %add : f32
  }
  return %ret : tensor<2x3xf32>
}
// CHECK-LABEL: @atomic_rmw
// CHECK: xla_gpu.atomic_rmw

// -----

func.func private @add(%a: f32, %b: f32) -> f32 {
  %ret = arith.addf %a, %b : f32
  return %ret : f32
}

func.func @caller(%a: f32, %b: f32) -> f32 {
  %c = xla_gpu.pure_call @add(%a, %b) : (f32, f32) -> (f32)
  %d = xla_gpu.pure_call @add(%a, %b) : (f32, f32) -> (f32)
  %ret = arith.addf %c, %d : f32
  return %ret : f32
}
// CHECK-LABEL: @caller
// CHECK: %[[C:.*]] = xla_gpu.pure_call @add
// CHECK: %[[D:.*]] = xla_gpu.pure_call @add
// CHECK: arith.addf %[[C]], %[[D]]

// CHECK-CSE: @caller
// CHECK-CSE: %[[C:.*]] = xla_gpu.pure_call @add
// CHECK-CSE: arith.addf %[[C]], %[[C]]

// -----

#map0 = affine_map<(d0, d1)[s0] -> (d0, d1 + s0)>
func.func @apply_indexing(%d0: index, %d1: index, %s0: index) -> (index, index) {
  %0:2 = xla_gpu.apply_indexing #map0 (%d0 in [0, 3), %d1 in [1, 4))[%s0 in [2, 5)]
  func.return %0#0, %0#1 : index, index
}
// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1)[s0] -> (d0, d1 + s0)>

// CHECK-LABEL: @apply_indexing
// CHECK: (%[[d0:.*]]: index, %[[d1:.*]]: index, %[[s0:.*]]: index)
// CHECK: xla_gpu.apply_indexing #[[$MAP0]]
// CHECK-SAME:  (%[[d0]] in [0, 3), %[[d1]] in [1, 4))[%[[s0]] in [2, 5)]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func.func @apply_indexing_no_symbols(%d0: index, %d1: index) -> (index, index) {
  %0:2 = xla_gpu.apply_indexing #map0 (%d0 in [0, 3), %d1 in [1, 4))
  func.return %0#0, %0#1 : index, index
}
// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @apply_indexing_no_symbols
// CHECK: (%[[d0:.*]]: index, %[[d1:.*]]: index)
// CHECK: xla_gpu.apply_indexing #[[$MAP0]]
// CHECK-SAME:  (%[[d0]] in [0, 3), %[[d1]] in [1, 4))

// -----

#map0 = affine_map<()[s0] -> (s0, s0)>
func.func @apply_indexing_no_dims(%s0: index) -> (index, index) {
  %0:2 = xla_gpu.apply_indexing #map0 [%s0 in [2, 5)]
  func.return %0#0, %0#1 : index, index
}
// CHECK: #[[$MAP0:.*]] = affine_map<()[s0] -> (s0, s0)>

// CHECK-LABEL: @apply_indexing_no_dims
// CHECK: (%[[s0:.*]]: index)
// CHECK: xla_gpu.apply_indexing #[[$MAP0]][%[[s0]] in [2, 5)]
