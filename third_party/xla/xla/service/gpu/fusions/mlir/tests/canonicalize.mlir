// RUN: mlir_fusions_opt %s --split-input-file -canonicalize | FileCheck %s

#map0 = affine_map<()[s0, s1] -> (1 + s0 + s1 mod 3 - s1, s0 mod 2)>
func.func @apply_indexing_no_dims(%s0: index, %s1: index) -> (index, index) {
  %0:2 = xla_gpu.apply_indexing #map0 [%s0 in [-10, 10], %s1 in [0, 2]]
  func.return %0#0, %0#1 : index, index
}
// CHECK: #[[$MAP:.*]] = affine_map<()[s0] -> (s0 + 1, s0 mod 2)>

// CHECK-LABEL: func.func @apply_indexing_no_dims
// CHECK-SAME:      %[[ARG_0:.*]]: index, %[[ARG_1:.*]]: index)
// CHECK:         xla_gpu.apply_indexing #[[$MAP]][%[[ARG_0]] in [-10, 10]]

// -----

#map0 = affine_map<(d0, d1)[s0] -> (d0 + s0, 4, d1, 1, s0)>
func.func @fold_indexing_map_results(%d0: index, %d1: index, %s0: index)
    -> (index, index, index, index, index) {
  %0:5 = xla_gpu.apply_indexing #map0 (%d0 in [-10, 10], %d1 in [0, 2])[%s0 in [-1, 1]]
  func.return %0#0, %0#1, %0#2, %0#3, %0#4  : index, index, index, index, index
}
// CHECK: #[[$MAP:.*]] = affine_map<(d0, d1)[s0] -> (d0 + s0)>

// CHECK-LABEL: func.func @fold_indexing_map_results
// CHECK-SAME:  %[[ARG_0:.*]]: index, %[[ARG_1:.*]]: index, %[[ARG_2:.*]]: index)

// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index

// CHECK:       %[[NEW_RESULT:.*]] = xla_gpu.apply_indexing #[[$MAP]]
// CHECK:       return %[[NEW_RESULT]], %[[C4]], %[[ARG_1]], %[[C1]], %[[ARG_2]]