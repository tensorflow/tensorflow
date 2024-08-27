// RUN: mlir_fusions_opt %s -xla-gpu-lower-xla-gpu-to-scf | FileCheck %s
func.func private @exp(%p0: tensor<32x64xf32>, %i: index, %j: index) -> f32

#map = #xla_gpu.indexing_map<(d0, d1)[s0, s1] -> (d1*32+d0*2+s0, s1), domain: d0 in [0, 32], d1 in [0, 8], s0 in [0, 1], s1 in [0, 1]>
#map1 = #xla_gpu.indexing_map<(d0, d1)[s0, s1] -> (d0*2+s0, s1), domain: d0 in [0, 32], d1 in [0, 2], s0 in [0, 1], s1 in [0, 1]>
func.func @materialize(%input: tensor<32x64xf32>, %i: index, %j: index) -> !xla_gpu.indexed_vector<32x2x2xf32, #map1> {
  %cla = arith.constant 0 : index
  %0 = xla_gpu.materialize @exp(%input) at #map(%i, %j) : (tensor<32x64xf32>) -> !xla_gpu.indexed_vector<32x2x2xf32, #map1>
  func.return %0 : !xla_gpu.indexed_vector<32x2x2xf32, #map1>
}

// CHECK-DAG: #[[$MAP:.*]] = #xla_gpu.indexing_map<(d0, d1)[s0, s1] -> (d1 * 32 + d0 * 2 + s0, s1)
// CHECK-DAG: #[[$MAP1:.*]] = #xla_gpu.indexing_map<(d0, d1)[s0, s1] -> (d0 * 2 + s0, s1)
// CHECK: @materialize(%[[INPUT:.*]]: tensor<32x64xf32>, %[[INDEX1:.*]]: index, %[[INDEX2:.*]]: index)
// CHECK: %[[INIT_VEC:.*]] = arith.constant {{.*}} : vector<2x2xf32>
// CHECK: xla_gpu.loop (%[[INDEX1]], %[[INDEX2]])[%[[S0:.*]], %[[S1:.*]]]
// CHECK-SAME: -> (%[[MAP_RESULT1:.*]], %[[MAP_RESULT2:.*]]) in
// CHECK-SAME: #[[$MAP]] iter_args(%[[ITER_ARG:.*]] = %[[INIT_VEC]])
// CHECK: %[[PURE_CALL:.*]] = xla_gpu.pure_call @exp(%[[INPUT]], %[[MAP_RESULT1]], %[[MAP_RESULT2]])
// CHECK: %[[VECTOR_INDEX:.*]]:2 = xla_gpu.apply_indexing #[[$MAP1]](%[[INDEX1]], %[[INDEX2]])[%[[S0]], %[[S1]]]
// CHECK: vector.insert %[[PURE_CALL]], %[[ITER_ARG]] [%[[VECTOR_INDEX]]#0, %[[VECTOR_INDEX]]#1]
// CHECK xla_gpu.yield %{{.*}} : vector<2x2xf32>