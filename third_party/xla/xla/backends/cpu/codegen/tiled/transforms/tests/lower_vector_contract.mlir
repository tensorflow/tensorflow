// RUN: fusion_compiler_opt %s -xtile-cpu-lower-vector-contract -split-input-file | FileCheck %s

#map_kminor_lhs = affine_map<(d0, d1) -> (d1, d0)>
#map_kminor_rhs = affine_map<(d0, d1) -> (d0)>
#map_kminor_acc = affine_map<(d0, d1) -> (d1)>

func.func @lower_k_minor_contract_outer_product(%lhs: vector<8x16xf32>,
                                                %rhs: vector<16xf32>,
                                                %acc: vector<8xf32>) -> vector<8xf32> {
  %0 = vector.contract {
      indexing_maps = [#map_kminor_lhs, #map_kminor_rhs, #map_kminor_acc],
      iterator_types = ["reduction", "parallel"],
      kind = #vector.kind<add>}
    %lhs, %rhs, %acc : vector<8x16xf32>, vector<16xf32> into vector<8xf32>
  return %0 : vector<8xf32>
}
// CHECK-LABEL: @lower_k_minor_contract_outer_product
// CHECK: vector.transpose %{{.*}}, [1, 0] : vector<8x16xf32> to vector<16x8xf32>
// CHECK-COUNT-16: vector.fma {{.*}} : vector<8xf32>
// CHECK: return %{{.*}} : vector<8xf32>

// -----

#map_kmajor_lhs = affine_map<(d0, d1) -> (d0, d1)>
#map_kmajor_rhs = affine_map<(d0, d1) -> (d0)>
#map_kmajor_acc = affine_map<(d0, d1) -> (d1)>

func.func @lower_k_major_contract_outer_product(%lhs: vector<8x8xf32>,
                                                %rhs: vector<8xf32>,
                                                %acc: vector<8xf32>) -> vector<8xf32> {
  %0 = vector.contract {
      indexing_maps = [#map_kmajor_lhs, #map_kmajor_rhs, #map_kmajor_acc],
      iterator_types = ["reduction", "parallel"],
      kind = #vector.kind<add>}
    %lhs, %rhs, %acc : vector<8x8xf32>, vector<8xf32> into vector<8xf32>
  return %0 : vector<8xf32>
}
// CHECK-LABEL: @lower_k_major_contract_outer_product
// CHECK-NOT: vector.transpose
// CHECK-COUNT-8: vector.fma {{.*}} : vector<8xf32>
// CHECK: return %{{.*}} : vector<8xf32>
