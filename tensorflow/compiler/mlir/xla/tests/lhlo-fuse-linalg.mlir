// RUN: tf-opt -lhlo-fuse-linalg %s -o - | FileCheck %s

#map0 = (d0, d1) -> (d0, d1)
#pointwise_2d_trait = {indexing_maps = [#map0, #map0, #map0], n_loop_types = [2, 0, 0], n_views = [2, 1]}
func @fusion(%input0: memref<2x2xf32>, %input1: memref<2x2xf32>, %input2: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %temp_result = alloc() {temp = true} : memref<2x2xf32>
  linalg.generic #pointwise_2d_trait %input1, %input2, %temp_result {
  ^bb0(%input1_in: f32, %input2_in: f32, %temp_result_in: f32):
    %1 = addf %input1_in, %input2_in : f32
    linalg.yield %1 : f32
  } : memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>
  linalg.generic #pointwise_2d_trait %temp_result, %input0, %result {
  ^bb0(%temp_result_in: f32, %input0_in: f32, %result_in: f32):
    %1 = mulf %temp_result_in, %input0_in : f32
    linalg.yield %1 : f32
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
