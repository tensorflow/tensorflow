// RUN: mlir_fusions_opt %s --split-input-file -canonicalize | FileCheck %s

#map0 = affine_map<()[s0, s1] -> (1 + s0 + s1 mod 3 - s1, s0 mod 2)>
func.func @simplify_apply_indexing(%s0: index, %s1: index) -> (index, index) {
  %0:2 = xla_gpu.apply_indexing #map0 [%s0 in [-10, 10), %s1 in [0, 3)]
  func.return %0#0, %0#1 : index, index
}
// CHECK: #[[$MAP:.*]] = affine_map<()[s0] -> (s0 + 1, s0 mod 2)>

// CHECK-LABEL: func.func @simplify_apply_indexing
// CHECK-SAME:      %[[ARG_0:.*]]: index, %[[ARG_1:.*]]: index)
// CHECK:         xla_gpu.apply_indexing #[[$MAP]][%[[ARG_0]] in [-10, 10)]

// -----

#map0 = affine_map<(d0, d1, d2)[s0, s1] -> (1 + s0 + s1 mod 4 - s1, s0 mod 2, d0 + d2)>
func.func @simplify_apply_indexing_remove_dims(%d0: index, %d1: index,
    %d2: index, %s0: index, %s1: index) -> (index, index, index) {
  %0:3 = xla_gpu.apply_indexing #map0
    (%d0 in [0, 2), %d1 in [0, 3), %d2 in [0, 4))
    [%s0 in [-11, 11), %s1 in [0, 4)]
  func.return %0#0, %0#1, %0#2 : index, index, index
}
// CHECK: #[[$MAP:.*]] = affine_map<(d0, d1)[s0] -> (s0 + 1, s0 mod 2, d0 + d1)>

// CHECK-LABEL: func.func @simplify_apply_indexing_remove_dims
// CHECK-SAME:      %[[ARG_0:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:      %[[ARG_1:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:      %[[ARG_2:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:      %[[ARG_3:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:      %[[ARG_4:[a-zA-Z0-9_]+]]: index)
// CHECK:       xla_gpu.apply_indexing #[[$MAP]]
// CHECK-SAME:    (%[[ARG_0]] in [0, 2), %[[ARG_2]] in [0, 4))
// CHECK-SAME:    [%[[ARG_3]] in [-11, 11)]

// -----

#map0 = affine_map<(d0, d1)[s0] -> (d0 + s0, 4, d1, 1, s0)>
func.func @fold_indexing_map_results(%d0: index, %d1: index, %s0: index)
    -> (index, index, index, index, index) {
  %0:5 = xla_gpu.apply_indexing #map0 (%d0 in [-10, 10), %d1 in [0, 3))[%s0 in [-1, 2)]
  func.return %0#0, %0#1, %0#2, %0#3, %0#4  : index, index, index, index, index
}
// CHECK: #[[$MAP:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>

// CHECK-LABEL: func.func @fold_indexing_map_results
// CHECK-SAME:  %[[ARG_0:.*]]: index, %[[ARG_1:.*]]: index, %[[ARG_2:.*]]: index)

// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index

// CHECK:       %[[NEW_RESULT:.*]] = xla_gpu.apply_indexing #[[$MAP]]
// CHECK:       return %[[NEW_RESULT]], %[[C4]], %[[ARG_1]], %[[C1]], %[[ARG_2]]

// -----

#map0 = affine_map<(d0, d1)[s0] -> (d0 + s0, s0 + 4, d1 mod 2, 1 + d1, s0)>
func.func @remove_unused_results(%d0: index, %d1: index, %s0: index) -> (index) {
  %0:5 = xla_gpu.apply_indexing #map0 (%d0 in [-10, 10), %d1 in [0, 3))[%s0 in [-1, 2)]
  func.return %0#2 : index
}
// CHECK: #[[$MAP:.*]] = affine_map<(d0) -> (d0 mod 2)>

// CHECK-LABEL: func.func @remove_unused_results
// CHECK-SAME:  %[[ARG_0:.*]]: index, %[[ARG_1:.*]]: index, %[[ARG_2:.*]]: index)

// CHECK:       %[[NEW_RESULT:.*]] = xla_gpu.apply_indexing #[[$MAP]](%[[ARG_1]] in [0, 3))
// CHECK:       return %[[NEW_RESULT]]

// -----

#map0 = affine_map<(d0, d1)[s0, s1] -> (d0 + d1 + s0 + s1 mod 3)>
func.func @fold_operands(%d0: index) -> index {
  %d1 = arith.constant 1 : index
  %s0 = arith.constant 2 : index
  %s1 = arith.constant 3 : index
  %0 = xla_gpu.apply_indexing #map0 (%d0 in [0, 11), %d1 in [0, 6))
                                    [%s0 in [-10, 10), %s1 in [0, 5)]
  func.return %0 : index
}
// CHECK: #[[$MAP:.*]] = affine_map<(d0) -> (d0 + 3)>

// CHECK-LABEL: func.func @fold_operands
// CHECK-SAME:      %[[ARG_0:.*]]: index)
// CHECK:         xla_gpu.apply_indexing #[[$MAP]](%[[ARG_0]] in [0, 11))

// -----

func.func @fold_operands_and_results(%arg0: index, %arg1: index)
  -> (index, index) {
  %0:2 = xla_gpu.apply_indexing affine_map<(d0, d1) -> (0, d1)>
    (%arg0 in [0, 5), %arg1 in [0, 6))
  return %0#0, %0#1 : index, index
}

// CHECK-LABEL: func.func @fold_operands_and_results
// CHECK-SAME:      %[[ARG_0:.*]]: index, %[[ARG_1:.*]]: index)
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0
// CHECK-NEXT: return %[[C0]], %[[ARG_1]] : index, index

// -----

func.func @fold_sequence(%arg0: index, %arg1: index) -> index {
  %0 = xla_gpu.apply_indexing affine_map<(d0, d1) -> (d0 + d1)>
    (%arg0 in [0, 6), %arg1 in [0, 5))
  %1 = xla_gpu.apply_indexing affine_map<(d0) -> (d0 mod 100 + 42)>
    (%0 in [0, 10001))
  func.return %1 : index
}

// CHECK: #[[$MAP:.*]] = affine_map<(d0, d1) -> (d0 + d1 + 42)>
// CHECK-LABEL: func.func @fold_sequence
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
// CHECK-NEXT:  xla_gpu.apply_indexing #[[$MAP]]
// CHECK-SAME:      (%[[ARG0]] in [0, 6), %[[ARG1]] in [0, 5))

// -----

func.func @fold_sequence_sym(%arg0: index, %arg1: index) -> index {
  %0 = xla_gpu.apply_indexing affine_map<(d0, d1) -> (d0 + d1)>
    (%arg0 in [0, 6), %arg1 in [0, 5))
  %1 = xla_gpu.apply_indexing affine_map<()[s0] -> (s0 mod 100 + 42)>
    [%0 in [0, 10001)]
  func.return %1 : index
}

// CHECK: #[[$MAP:.*]] = affine_map<(d0, d1) -> (d0 + d1 + 42)>
// CHECK-LABEL: func.func @fold_sequence_sym
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
// CHECK-NEXT:  xla_gpu.apply_indexing #[[$MAP]]
// CHECK-SAME:      (%[[ARG0]] in [0, 6), %[[ARG1]] in [0, 5))

// -----

func.func @fold_sequence_shared_operands(%arg0: index, %arg1: index) -> index {
  %0 = xla_gpu.apply_indexing affine_map<(d0, d1) -> (d0 + d1)>
    (%arg0 in [0, 6), %arg1 in [0, 5))
  %1 = xla_gpu.apply_indexing affine_map<(d0, d1) -> (d0 + d1)>
    (%arg1 in [0, 5), %0 in [0, 10001))
  func.return %1 : index
}

// CHECK: #[[$MAP:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK-LABEL: func.func @fold_sequence_shared_operands
// CHECK-SAME:      %[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
// CHECK-NEXT:  xla_gpu.apply_indexing #[[$MAP]]
// CHECK-SAME:      (%[[ARG1]] in [0, 5), %[[ARG0]] in [0, 6))
