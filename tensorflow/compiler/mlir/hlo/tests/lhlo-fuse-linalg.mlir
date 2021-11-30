// RUN: mlir-hlo-opt -lhlo-fuse-linalg %s -split-input-file | FileCheck %s --dump-input=always
// RUN: mlir-hlo-opt -lhlo-fuse-linalg=tile-sizes=2,3 %s -split-input-file | FileCheck %s -check-prefix=TILED
// RUN: mlir-hlo-opt -lhlo-fuse-linalg=use-parallel-loops %s -split-input-file | FileCheck %s -check-prefix=PLOOP

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#pointwise_2d_trait = {indexing_maps = [#map0, #map0, #map0],
                       iterator_types = ["parallel", "parallel"]}
func @fusion(%multiplier: memref<6x6xf32>, %summand_1: memref<6x6xf32>,
             %summand_2: memref<6x6xf32>, %result: memref<6x6xf32>) {
  %temp_result = memref.alloc() : memref<6x6xf32>
  linalg.generic #pointwise_2d_trait
    ins(%summand_1, %summand_2 : memref<6x6xf32>, memref<6x6xf32>)
   outs(%temp_result : memref<6x6xf32>) {
  ^bb0(%summand_1_in: f32, %summand_2_in: f32, %temp_result_in: f32):
    %out = arith.addf %summand_1_in, %summand_2_in : f32
    linalg.yield %out : f32
  }
  linalg.generic #pointwise_2d_trait
    ins(%temp_result, %multiplier : memref<6x6xf32>, memref<6x6xf32>)
   outs(%result : memref<6x6xf32>) {
  ^bb0(%temp_result_in: f32, %multiplier_in: f32, %result_in: f32):
    %out = arith.mulf %temp_result_in, %multiplier_in : f32
    linalg.yield %out : f32
  }
  memref.dealloc %temp_result : memref<6x6xf32>
  return
}
// CHECK-LABEL: func @fusion
//       CHECK:  %[[C1:.*]] = arith.constant 1
//   CHECK-NOT:  linalg.generic
//       CHECK:  scf.for {{.*}} step %[[C1]]
//       CHECK:    scf.for {{.*}} step %[[C1]]
//   CHECK-NOT:  scf.for
//       CHECK:      linalg.generic
//       CHECK:        addf
//       CHECK:      linalg.generic
//       CHECK:        mulf

// TILED-LABEL: func @fusion
//   TILED-DAG:  %[[C2:.*]] = arith.constant 2
//   TILED-DAG:  %[[C3:.*]] = arith.constant 3
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
 %0 = memref.alloc() : memref<100x10xf32>
 linalg.generic {
   indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                    affine_map<(d0, d1) -> (d0, d1)>],
   iterator_types = ["parallel", "parallel"]}
     ins(%arg1 : memref<100xf32>)
    outs(%0 : memref<100x10xf32>) {
   ^bb0(%arg3: f32, %arg4: f32): // no predecessors
     linalg.yield %arg3 : f32
   }
 %1 = memref.alloc() : memref<100x10xf32>
 linalg.generic {
   indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                    affine_map<(d0, d1) -> (d0, d1)>,
                    affine_map<(d0, d1) -> (d0, d1)>],
   iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %0 : memref<100x10xf32>, memref<100x10xf32>)
   outs(%1 : memref<100x10xf32>) {
     ^bb0(%arg3: f32, %arg4: f32, %arg5: f32): // no predecessors
       %2 = arith.subf %arg3, %arg4 : f32
       linalg.yield %2 : f32
     }
 memref.dealloc %0 : memref<100x10xf32>
 linalg.generic {
   indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                    affine_map<(d0, d1) -> (d0, d1)>],
   iterator_types = ["parallel", "parallel"]}
     ins(%1 : memref<100x10xf32>)
    outs(%arg2 : memref<100x10xf32>) {
     ^bb0(%arg3: f32, %arg4: f32): // no predecessors
       %2 = math.exp %arg3 : f32
       linalg.yield %2 : f32
     }
 memref.dealloc %1 : memref<100x10xf32>
 return
}
// CHECK-LABEL: func @fusion
//       CHECK:  %[[C1:.*]] = arith.constant 1 :
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
//   TILED-DAG:   %[[C2:.*]] = arith.constant 2
//   TILED-DAG:   %[[C3:.*]] = arith.constant 3
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
#pointwise_4d_trait = {indexing_maps = [#map0, #map0, #map0],
                       iterator_types = ["parallel", "parallel", "parallel",
                                         "parallel"]}
func @fusion_4d(%multiplier: memref<6x6x6x6xf32>, %summand_1: memref<6x6x6x6xf32>,
             %summand_2: memref<6x6x6x6xf32>, %result: memref<6x6x6x6xf32>) {
  %temp_result = memref.alloc() : memref<6x6x6x6xf32>
  linalg.generic #pointwise_4d_trait
    ins(%summand_1, %summand_2 : memref<6x6x6x6xf32>, memref<6x6x6x6xf32>)
   outs(%temp_result : memref<6x6x6x6xf32>) {
  ^bb0(%summand_1_in: f32, %summand_2_in: f32, %temp_result_in: f32):
    %out = arith.addf %summand_1_in, %summand_2_in : f32
    linalg.yield %out : f32
  }
  linalg.generic #pointwise_4d_trait
    ins(%temp_result, %multiplier : memref<6x6x6x6xf32>, memref<6x6x6x6xf32>)
   outs(%result : memref<6x6x6x6xf32>) {
  ^bb0(%temp_result_in: f32, %multiplier_in: f32, %result_in: f32):
    %out = arith.mulf %temp_result_in, %multiplier_in : f32
    linalg.yield %out : f32
  }
  memref.dealloc %temp_result : memref<6x6x6x6xf32>
  return
}
// CHECK-LABEL: func @fusion_4d
//       CHECK:  %[[C1:.*]] = arith.constant 1
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
//   TILED-DAG:  %[[C2:.*]] = arith.constant 2
//   TILED-DAG:  %[[C3:.*]] = arith.constant 3
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
#pointwise_2d_trait = {indexing_maps = [#map0, #map0, #map0],
                       iterator_types = ["parallel", "parallel"]}
func @fusion(%multiplier: memref<6x6xf32>, %summand_1: memref<6x6xf32>,
             %summand_2: memref<6x6xf32>) -> memref<6x6xf32> {
  %temp_result = memref.alloc() : memref<6x6xf32>
  linalg.generic #pointwise_2d_trait
    ins(%summand_1, %summand_2 : memref<6x6xf32>, memref<6x6xf32>)
   outs(%temp_result : memref<6x6xf32>) {
  ^bb0(%summand_1_in: f32, %summand_2_in: f32, %temp_result_in: f32):
    %out = arith.addf %summand_1_in, %summand_2_in : f32
    linalg.yield %out : f32
  }
  %result = memref.alloc() : memref<6x6xf32>
  linalg.generic #pointwise_2d_trait
    ins(%temp_result, %multiplier : memref<6x6xf32>, memref<6x6xf32>)
   outs(%result : memref<6x6xf32>) {
  ^bb0(%temp_result_in: f32, %multiplier_in: f32, %result_in: f32):
    %out = arith.mulf %temp_result_in, %multiplier_in : f32
    linalg.yield %out : f32
  }
  memref.dealloc %temp_result : memref<6x6xf32>
  return %result : memref<6x6xf32>
}

// CHECK-LABEL: func @fusion
//       CHECK:  %[[C1:.*]] = arith.constant 1
//   CHECK-NOT:  linalg.generic
//       CHECK:  scf.for {{.*}} step %[[C1]]
//       CHECK:    scf.for {{.*}} step %[[C1]]
//   CHECK-NOT:  scf.for
//       CHECK:      linalg.generic
//       CHECK:        addf
//       CHECK:      linalg.generic
//       CHECK:        mulf

// TILED-LABEL: func @fusion
//   TILED-DAG:  %[[C2:.*]] = arith.constant 2
//   TILED-DAG:  %[[C3:.*]] = arith.constant 3
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

func @view_result(%arg0: memref<?xf32>, %arg1: memref<?xindex>, %arg2: index)
    -> memref<*xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %1 = memref.alloc(%arg2) : memref<?xf32>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                   affine_map<(d0) -> (d0)>],
                  iterator_types = ["parallel"]}
      ins(%arg0 : memref<?xf32>) outs(%1 : memref<?xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
    %13 = math.abs %arg3 : f32
    linalg.yield %13 : f32
  }
  %2 = memref.reshape %1(%arg1)
      : (memref<?xf32>, memref<?xindex>) -> memref<*xf32>
  return %2 : memref<*xf32>
}

// CHECK-LABEL: func @view_result
//       CHECK:  %[[C1:.*]] = arith.constant 1
//   CHECK-NOT:  linalg.generic
//       CHECK:  scf.for {{.*}} step %[[C1]]
//   CHECK-NOT:  scf.for
//       CHECK:      linalg.generic
//       CHECK:        math.abs
//       CHECK:  memref.reshape

// TILED-LABEL: func @view_result
//   TILED-DAG:  %[[C2:.*]] = arith.constant 2
//   TILED-NOT:  linalg.generic
//       TILED:  scf.for {{.*}} step %[[C2]]
//   TILED-NOT:  scf.for
//       TILED:      linalg.generic
//       TILED:        math.abs
//       TILED:  memref.reshape


// PLOOP-LABEL: func @view_result
//   PLOOP-NOT:  linalg.generic
//       PLOOP:  scf.parallel
//   PLOOP-NOT:  scf.parallel
//       PLOOP:      linalg.generic
//       PLOOP:        math.abs
//       PLOOP:  memref.reshape



// -----

// Confirm that tiling information is passed through RegionBranchOpInterfaces.
// This test also uses memref.reshape, just to have a value to return through
// the if statement.
func @branching_result(%arg0: memref<?xf32>, %arg1: memref<?xindex>, %arg2: index)
    -> memref<*xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %1 = memref.alloc(%arg2) : memref<?xf32>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                   affine_map<(d0) -> (d0)>],
                  iterator_types = ["parallel"]}
      ins(%arg0 : memref<?xf32>) outs(%1 : memref<?xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
    %13 = math.abs %arg3 : f32
    linalg.yield %13 : f32
  }
  %true = arith.constant 1 : i1
  %3 = scf.if %true -> memref<*xf32> {
    %2 = memref.reshape %1(%arg1)
        : (memref<?xf32>, memref<?xindex>) -> memref<*xf32>
    scf.yield %2 : memref<*xf32>
  } else {
    %2 = memref.reshape %1(%arg1)
        : (memref<?xf32>, memref<?xindex>) -> memref<*xf32>
    scf.yield %2 : memref<*xf32>
  }
  return %3 : memref<*xf32>
}

// CHECK-LABEL: func @branching_result
//       CHECK:  %[[C1:.*]] = arith.constant 1
//   CHECK-NOT:  linalg.generic
//       CHECK:  scf.for {{.*}} step %[[C1]]
//   CHECK-NOT:  scf.for
//       CHECK:      linalg.generic
//       CHECK:        math.abs
//       CHECK:  scf.if
//       CHECK:    memref.reshape
//       CHECK:    scf.yield
//       CHECK:  else
//       CHECK:    memref.reshape
//       CHECK:    scf.yield

// TILED-LABEL: func @branching_result
//   TILED-DAG:  %[[C2:.*]] = arith.constant 2
//   TILED-NOT:  linalg.generic
//       TILED:  scf.for {{.*}} step %[[C2]]
//   TILED-NOT:  scf.for
//       TILED:      linalg.generic
//       TILED:        math.abs
//       TILED:  scf.if
//       TILED:    memref.reshape
//       TILED:    scf.yield
//       TILED:  else
//       TILED:    memref.reshape
//       TILED:    scf.yield

// PLOOP-LABEL: func @branching_result
//   PLOOP-NOT:  linalg.generic
//       PLOOP:  scf.parallel
//   PLOOP-NOT:  scf.parallel
//       PLOOP:      linalg.generic
//       PLOOP:        math.abs
//       PLOOP:  scf.if
//       PLOOP:    memref.reshape
//       PLOOP:    scf.yield
//       PLOOP:  else
//       PLOOP:    memref.reshape
//       PLOOP:    scf.yield

// -----

// Confirm that tiling information is passed through tensor_load, tensor.cast
// and memref_to_tensor  operations.
func @tensor_ops(%arg0: memref<32xf32>, %arg1: memref<32xindex>)
    -> memref<?xf32> {
  %c1 = arith.constant 1 : index
  %1 = memref.alloc() : memref<32xf32>
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                   affine_map<(d0) -> (d0)>],
                  iterator_types = ["parallel"]}
      ins(%arg0 : memref<32xf32>) outs(%1 : memref<32xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
    %13 = math.abs %arg3 : f32
    linalg.yield %13 : f32
  }
  %2 = bufferization.to_tensor %1 : memref<32xf32>
  %3 = tensor.cast %2 : tensor<32xf32> to tensor<?xf32>
  %4 = bufferization.to_memref %3 : memref<?xf32>
  return %4 : memref<?xf32>
}

// CHECK-LABEL: func @tensor_ops
//       CHECK:  %[[C1:.*]] = arith.constant 1
//   CHECK-NOT:  linalg.generic
//       CHECK:  scf.for {{.*}} step %[[C1]]
//   CHECK-NOT:  scf.for
//       CHECK:      linalg.generic
//       CHECK:        math.abs
//       CHECK:  bufferization.to_tensor
//       CHECK:  tensor.cast
//       CHECK:  bufferization.to_memref

// TILED-LABEL: func @tensor_ops
//   TILED-DAG:  %[[C2:.*]] = arith.constant 2
//   TILED-NOT:  linalg.generic
//       TILED:  scf.for {{.*}} step %[[C2]]
//   TILED-NOT:  scf.for
//       TILED:      linalg.generic
//       TILED:        math.abs
//       TILED:  bufferization.to_tensor
//       TILED:  tensor.cast
//       TILED:  bufferization.to_memref


// PLOOP-LABEL: func @tensor_ops
//   PLOOP-NOT:  linalg.generic
//       PLOOP:  scf.parallel
//   PLOOP-NOT:  scf.parallel
//       PLOOP:      linalg.generic
//       PLOOP:        math.abs
//       PLOOP:  bufferization.to_tensor
//       PLOOP:  tensor.cast
//       PLOOP:  bufferization.to_memref
