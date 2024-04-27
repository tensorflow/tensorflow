// RUN: mlir_fusions_opt %s --split-input-file -canonicalize | FileCheck %s

#map0 = affine_map<()[s0, s1] -> (s0 + s1 mod 3 - s1, s0)>
func.func @apply_indexing_no_dims(%s0: index, %s1: index) -> (index, index) {
  %0:2 = xla_gpu.apply_indexing #map0 [%s0 in [-10, 10], %s1 in [0, 2]]
  func.return %0#0, %0#1 : index, index
}
// CHECK: #[[$MAP:.*]] = affine_map<()[s0] -> (s0, s0)>

// CHECK-LABEL: func.func @apply_indexing_no_dims
// CHECK-SAME:      %[[ARG_0:.*]]: index, %[[ARG_1:.*]]: index)
// CHECK:         xla_gpu.apply_indexing #[[$MAP]][%[[ARG_0]] in [-10, 10]]