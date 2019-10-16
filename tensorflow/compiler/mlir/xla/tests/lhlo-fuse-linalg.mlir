// RUN: tf-opt -lhlo-fuse-linalg %s -o - | FileCheck %s

#map0 = (d0, d1) -> (d0, d1)
#pointwise_2d_trait = {indexing_maps = [#map0, #map0, #map0], n_loop_types = [2, 0, 0], n_views = [2, 1]}
func @fusion(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>, %arg3: memref<2x2xf32>) {
  %0 = alloc() {temp = true} : memref<2x2xf32>
  linalg.generic #pointwise_2d_trait %arg1, %arg2, %0 {
  ^bb0(%block_arg0: f32, %block_arg1: f32, %block_arg2: f32):
    %1 = addf %block_arg0, %block_arg1 : f32
    linalg.yield %1 : f32
  } : memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>
  linalg.generic #pointwise_2d_trait %0, %arg0, %arg3 {
  ^bb0(%block_arg0: f32, %block_arg1: f32, %block_arg2: f32):
    %1 = mulf %block_arg0, %block_arg1 : f32
    linalg.yield %1 : f32
  } : memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>
  dealloc %0 : memref<2x2xf32>
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
