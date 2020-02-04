// RUN: tf-opt -lhlo-fuse-linalg %s -o - | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#pointwise_2d_trait = {args_in = 2, args_out = 1, indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
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

func @fusion_of_three(%arg0: memref<100x10xf32>,
                      %arg1: memref<100xf32>,
                      %arg2: memref<100x10xf32>) {
 %0 = alloc() {temp = true} : memref<100x10xf32>
 linalg.generic {
   args_in = 1 : i64,
   args_out = 1 : i64,
   indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
   iterator_types = ["parallel", "parallel"]
 } %arg1, %0 {
     ^bb0(%arg3: f32, %arg4: f32): // no predecessors
       linalg.yield %arg3 : f32
     }: memref<100xf32>, memref<100x10xf32>
 %1 = alloc() {temp = true} : memref<100x10xf32>
 linalg.generic {
   args_in = 2 : i64,
   args_out = 1 : i64,
   indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
   iterator_types = ["parallel", "parallel"]
 } %arg0, %0, %1 {
     ^bb0(%arg3: f32, %arg4: f32, %arg5: f32): // no predecessors
       %2 = subf %arg3, %arg4 : f32
       linalg.yield %2 : f32
     }: memref<100x10xf32>, memref<100x10xf32>, memref<100x10xf32>
 dealloc %0 : memref<100x10xf32>
 linalg.generic {
   args_in = 1 : i64,
   args_out = 1 : i64,
   indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
   iterator_types = ["parallel", "parallel"]
 } %1, %arg2 {
     ^bb0(%arg3: f32, %arg4: f32): // no predecessors
       %2 = exp %arg3 : f32
       linalg.yield %2 : f32
     }: memref<100x10xf32>, memref<100x10xf32>
 dealloc %1 : memref<100x10xf32>
 return
}
// CHECK-LABEL: func @fusion
//   CHECK-NOT:   linalg.generic
//       CHECK:   loop.for
//       CHECK:     loop.for
//   CHECK-NOT:   loop.for
//       CHECK:       linalg.generic
//       CHECK:       linalg.generic
//       CHECK:         subf
//       CHECK:       linalg.generic
//       CHECK:         exp
