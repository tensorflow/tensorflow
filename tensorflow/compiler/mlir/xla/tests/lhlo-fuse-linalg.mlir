// RUN: xla-opt -lhlo-fuse-linalg %s -split-input-file | FileCheck %s --dump-input=always
// RUN: xla-opt -lhlo-fuse-linalg=tile-sizes=2,3 %s -split-input-file | FileCheck %s -check-prefix=TILED
// RUN: xla-opt -lhlo-fuse-linalg=use-parallel-loops %s -split-input-file | FileCheck %s -check-prefix=PLOOP

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#pointwise_2d_trait = {args_in = 2, args_out = 1, indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
func @fusion(%multiplier: memref<6x6xf32>, %summand_1: memref<6x6xf32>,
             %summand_2: memref<6x6xf32>, %result: memref<6x6xf32>) {
  %temp_result = alloc() : memref<6x6xf32>
  linalg.generic #pointwise_2d_trait %summand_1, %summand_2, %temp_result {
  ^bb0(%summand_1_in: f32, %summand_2_in: f32, %temp_result_in: f32):
    %out = addf %summand_1_in, %summand_2_in : f32
    linalg.yield %out : f32
  } : memref<6x6xf32>, memref<6x6xf32>, memref<6x6xf32>
  linalg.generic #pointwise_2d_trait %temp_result, %multiplier, %result {
  ^bb0(%temp_result_in: f32, %multiplier_in: f32, %result_in: f32):
    %out = mulf %temp_result_in, %multiplier_in : f32
    linalg.yield %out : f32
  } : memref<6x6xf32>, memref<6x6xf32>, memref<6x6xf32>
  dealloc %temp_result : memref<6x6xf32>
  return
}
// CHECK-LABEL: func @fusion
//       CHECK:  %[[C1:.*]] = constant 1
//   CHECK-NOT:  linalg.generic
//       CHECK:  scf.for {{.*}} step %[[C1]]
//       CHECK:    scf.for {{.*}} step %[[C1]]
//   CHECK-NOT:  scf.for
//       CHECK:      linalg.generic
//       CHECK:        addf
//       CHECK:      linalg.generic
//       CHECK:        mulf

// TILED-LABEL: func @fusion
//   TILED-DAG:  %[[C2:.*]] = constant 2
//   TILED-DAG:  %[[C3:.*]] = constant 3
//   TILED-NOT:  linalg.generic
//       TILED:  scf.for {{.*}} step %[[C2]]
//       TILED:    scf.for {{.*}} step %[[C3]]
//   TILED-NOT:  scf.for
//       TILED:      linalg.generic
//       TILED:        addf
//       TILED:      linalg.generic
//       TILED:        mulf

// PLOOP-LABEL: func @fusion
//   PLOOP-NOT:  linalg.generic
//       PLOOP:  scf.parallel
//   PLOOP-NOT:  scf.parallel
//       PLOOP:      linalg.generic
//       PLOOP:        addf
//       PLOOP:      linalg.generic
//       PLOOP:        mulf

// -----

func @fusion_of_three(%arg0: memref<100x10xf32>,
                      %arg1: memref<100xf32>,
                      %arg2: memref<100x10xf32>) {
 %0 = alloc() : memref<100x10xf32>
 linalg.generic {
   args_in = 1 : i64,
   args_out = 1 : i64,
   indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
   iterator_types = ["parallel", "parallel"]
 } %arg1, %0 {
     ^bb0(%arg3: f32, %arg4: f32): // no predecessors
       linalg.yield %arg3 : f32
     }: memref<100xf32>, memref<100x10xf32>
 %1 = alloc() : memref<100x10xf32>
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
//       CHECK:  %[[C1:.*]] = constant 1
//   CHECK-NOT:  linalg.generic
//       CHECK:  scf.for {{.*}} step %[[C1]]
//       CHECK:    scf.for {{.*}} step %[[C1]]
//   CHECK-NOT:  scf.for
//       CHECK:      linalg.generic
//       CHECK:      linalg.generic
//       CHECK:        subf
//       CHECK:      linalg.generic
//       CHECK:        exp

// TILED-LABEL: func @fusion_of_three
//   TILED-DAG:   %[[C2:.*]] = constant 2
//   TILED-DAG:   %[[C3:.*]] = constant 3
//   TILED-NOT:   linalg.generic
//       TILED:   scf.for {{.*}} step %[[C2]]
//       TILED:     scf.for {{.*}} step %[[C3]]
//   TILED-NOT:   scf.for
//       TILED:       linalg.generic
//       TILED:       linalg.generic
//       TILED:         subf
//       TILED:       linalg.generic
//       TILED:         exp

// PLOOP-LABEL: func @fusion_of_three
//   PLOOP-NOT:   linalg.generic
//       PLOOP:   scf.parallel
//   PLOOP-NOT:   scf.parallel
//       PLOOP:       linalg.generic
//       PLOOP:       linalg.generic
//       PLOOP:         subf
//       PLOOP:       linalg.generic
//       PLOOP:         exp

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#pointwise_4d_trait = {args_in = 2, args_out = 1, indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
func @fusion_4d(%multiplier: memref<6x6x6x6xf32>, %summand_1: memref<6x6x6x6xf32>,
             %summand_2: memref<6x6x6x6xf32>, %result: memref<6x6x6x6xf32>) {
  %temp_result = alloc() : memref<6x6x6x6xf32>
  linalg.generic #pointwise_4d_trait %summand_1, %summand_2, %temp_result {
  ^bb0(%summand_1_in: f32, %summand_2_in: f32, %temp_result_in: f32):
    %out = addf %summand_1_in, %summand_2_in : f32
    linalg.yield %out : f32
  } : memref<6x6x6x6xf32>, memref<6x6x6x6xf32>, memref<6x6x6x6xf32>
  linalg.generic #pointwise_4d_trait %temp_result, %multiplier, %result {
  ^bb0(%temp_result_in: f32, %multiplier_in: f32, %result_in: f32):
    %out = mulf %temp_result_in, %multiplier_in : f32
    linalg.yield %out : f32
  } : memref<6x6x6x6xf32>, memref<6x6x6x6xf32>, memref<6x6x6x6xf32>
  dealloc %temp_result : memref<6x6x6x6xf32>
  return
}
// CHECK-LABEL: func @fusion_4d
//       CHECK:  %[[C1:.*]] = constant 1
//   CHECK-NOT:  linalg.generic
//       CHECK:  scf.for {{.*}} step %[[C1]]
//       CHECK:    scf.for {{.*}} step %[[C1]]
//       CHECK:      scf.for {{.*}} step %[[C1]]
//       CHECK:        scf.for {{.*}} step %[[C1]]
//   CHECK-NOT:  scf.for
//       CHECK:      linalg.generic
//       CHECK:        addf
//       CHECK:      linalg.generic
//       CHECK:        mulf

// TILED-LABEL: func @fusion_4d
//   TILED-DAG:  %[[C2:.*]] = constant 2
//   TILED-DAG:  %[[C3:.*]] = constant 3
//   TILED-NOT:  linalg.generic
//       TILED:  scf.for {{.*}} step %[[C2]]
//       TILED:    scf.for {{.*}} step %[[C3]]
//   TILED-NOT:  scf.for
//       TILED:      linalg.generic
//       TILED:        addf
//       TILED:      linalg.generic
//       TILED:        mulf

// PLOOP-LABEL: func @fusion_4d
//   PLOOP-NOT:  linalg.generic
//       PLOOP:  scf.parallel
//   PLOOP-NOT:  scf.parallel
//       PLOOP:      linalg.generic
//       PLOOP:        addf
//       PLOOP:      linalg.generic
//       PLOOP:        mulf

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#pointwise_2d_trait = {args_in = 2, args_out = 1, indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
func @fusion(%multiplier: memref<6x6xf32>, %summand_1: memref<6x6xf32>,
             %summand_2: memref<6x6xf32>) -> memref<6x6xf32> {
  %temp_result = alloc() : memref<6x6xf32>
  linalg.generic #pointwise_2d_trait %summand_1, %summand_2, %temp_result {
  ^bb0(%summand_1_in: f32, %summand_2_in: f32, %temp_result_in: f32):
    %out = addf %summand_1_in, %summand_2_in : f32
    linalg.yield %out : f32
  } : memref<6x6xf32>, memref<6x6xf32>, memref<6x6xf32>
  %result = alloc() : memref<6x6xf32>
  linalg.generic #pointwise_2d_trait %temp_result, %multiplier, %result {
  ^bb0(%temp_result_in: f32, %multiplier_in: f32, %result_in: f32):
    %out = mulf %temp_result_in, %multiplier_in : f32
    linalg.yield %out : f32
  } : memref<6x6xf32>, memref<6x6xf32>, memref<6x6xf32>
  dealloc %temp_result : memref<6x6xf32>
  return %result : memref<6x6xf32>
}

// CHECK-LABEL: func @fusion
//       CHECK:  %[[C1:.*]] = constant 1
//   CHECK-NOT:  linalg.generic
//       CHECK:  scf.for {{.*}} step %[[C1]]
//       CHECK:    scf.for {{.*}} step %[[C1]]
//   CHECK-NOT:  scf.for
//       CHECK:      linalg.generic
//       CHECK:        addf
//       CHECK:      linalg.generic
//       CHECK:        mulf

// TILED-LABEL: func @fusion
//   TILED-DAG:  %[[C2:.*]] = constant 2
//   TILED-DAG:  %[[C3:.*]] = constant 3
//   TILED-NOT:  linalg.generic
//       TILED:  scf.for {{.*}} step %[[C2]]
//       TILED:    scf.for {{.*}} step %[[C3]]
//   TILED-NOT:  scf.for
//       TILED:      linalg.generic
//       TILED:        addf
//       TILED:      linalg.generic
//       TILED:        mulf

// PLOOP-LABEL: func @fusion
//   PLOOP-NOT:  linalg.generic
//       PLOOP:  scf.parallel
//   PLOOP-NOT:  scf.parallel
//       PLOOP:      linalg.generic
//       PLOOP:        addf
//       PLOOP:      linalg.generic
//       PLOOP:        mulf
