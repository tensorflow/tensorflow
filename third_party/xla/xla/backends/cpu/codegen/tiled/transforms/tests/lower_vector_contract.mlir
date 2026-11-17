// RUN: fusion_compiler_opt %s -xtile-cpu-lower-vector-contract -split-input-file | FileCheck %s

#map_lhs = affine_map<(d0, d1, d2) -> (d1, d0)>
#map_rhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_acc = affine_map<(d0, d1, d2) -> (d1, d2)>

func.func @drop_unit_rhs_parallel(%lhs: vector<8x16xf32>,
                                  %rhs: vector<16x1xf32>,
                                  %acc: vector<8x1xf32>) -> vector<8x1xf32> {
  %0 = vector.contract {
      indexing_maps = [#map_lhs, #map_rhs, #map_acc],
      iterator_types = ["reduction", "parallel", "parallel"],
      kind = #vector.kind<add>}
    %lhs, %rhs, %acc : vector<8x16xf32>, vector<16x1xf32> into vector<8x1xf32>
  return %0 : vector<8x1xf32>
}
// CHECK-LABEL: @drop_unit_rhs_parallel
// CHECK: vector.transpose %{{.*}}, [1, 0] : vector<8x16xf32> to vector<16x8xf32>
// CHECK-COUNT-16: vector.fma {{.*}} : vector<8xf32>
// CHECK: %[[OUT:.*]] = vector.shape_cast %{{.*}} : vector<8xf32> to vector<8x1xf32>
// CHECK: return %[[OUT]] : vector<8x1xf32>

// -----

#map_b_lhs = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>
#map_b_rhs = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map_b_acc = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>

func.func @drop_unit_batch_and_free_bf16(%lhs: vector<1x8x8xbf16>,
                                         %rhs: vector<1x8x1xbf16>,
                                         %acc: vector<1x8x1xf32>) -> vector<1x8x1xf32> {
  %0 = vector.contract {
      indexing_maps = [#map_b_lhs, #map_b_rhs, #map_b_acc],
      iterator_types = ["parallel", "reduction", "parallel", "parallel"],
      kind = #vector.kind<add>}
    %lhs, %rhs, %acc : vector<1x8x8xbf16>, vector<1x8x1xbf16> into vector<1x8x1xf32>
  return %0 : vector<1x8x1xf32>
}
// CHECK-LABEL: @drop_unit_batch_and_free_bf16
// CHECK: arith.extf %{{.*}} : vector<1x8x8xbf16> to vector<1x8x8xf32>
// CHECK: arith.extf %{{.*}} : vector<1x8x1xbf16> to vector<1x8x1xf32>
// CHECK: vector.transpose %{{.*}}, [1, 0] : vector<8x8xf32> to vector<8x8xf32>
// CHECK-COUNT-8: vector.fma {{.*}} : vector<8xf32>

// -----

#map_0d_lhs = affine_map<(d0, d1, d2) -> (d1, d0)>
#map_0d_rhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_0d_acc = affine_map<(d0, d1, d2) -> (d1, d2)>

func.func @collapse_all_unit_parallel_to_reduction(
    %lhs: vector<1x16xf32>, %rhs: vector<16x1xf32>, %acc: vector<1x1xf32>) -> vector<1x1xf32> {
  %0 = vector.contract {
      indexing_maps = [#map_0d_lhs, #map_0d_rhs, #map_0d_acc],
      iterator_types = ["reduction", "parallel", "parallel"],
      kind = #vector.kind<add>}
    %lhs, %rhs, %acc : vector<1x16xf32>, vector<16x1xf32> into vector<1x1xf32>
  return %0 : vector<1x1xf32>
}
// CHECK-LABEL: @collapse_all_unit_parallel_to_reduction
// CHECK: %[[MUL:.*]] = arith.mulf %{{.*}}, %{{.*}} : vector<16xf32>
// CHECK: %[[ACC:.*]] = vector.extract %{{.*}}[0, 0] : f32 from vector<1x1xf32>
// CHECK: %[[RED:.*]] = vector.reduction <add>, %[[MUL]], %[[ACC]]
// CHECK: %[[BCAST:.*]] = vector.broadcast %[[RED]] : f32 to vector<1x1xf32>
