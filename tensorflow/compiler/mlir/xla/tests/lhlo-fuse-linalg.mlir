// RUN: tf-opt -lhlo-fuse-linalg %s -o - | FileCheck %s

#map0 = (d0, d1) -> (d0, d1)
#pointwise_2d_trait = {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"], n_views = [2, 1]}
func @fusion(%multiplier: memref<2x2xf32>, %summand_1: memref<2x2xf32>,
             %summand_2: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %temp_result = alloc() {temp = true} : memref<2x2xf32>
  linalg.generic #pointwise_2d_trait %summand_1, %summand_2, %temp_result {
  ^bb0(%summand_1_in: f32, %summand_2_in: f32, %temp_result_in: f32):
    %out = addf %summand_1_in, %summand_2_in : f32
    linalg.yield %out : f32
  } : memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>
  linalg.generic #pointwise_2d_trait %temp_result, %multiplier, %result {
  ^bb0(%temp_result_in: f32, %multiplier_in: f32, %result_in: f32):
    %out = mulf %temp_result_in, %multiplier_in : f32
    linalg.yield %out : f32
  } : memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>
  dealloc %temp_result : memref<2x2xf32>
  "xla_lhlo.terminator"() : () -> ()
}

// CHECK-LABEL: func @fusion
//   CHECK-NOT:   linalg.generic
//       CHECK:   loop.for
//       CHECK:     loop.for
//   CHECK-NOT:   loop.for
//       CHECK:       linalg.generic
//       CHECK:         addf
//       CHECK:       linalg.generic
//       CHECK:         mulf
