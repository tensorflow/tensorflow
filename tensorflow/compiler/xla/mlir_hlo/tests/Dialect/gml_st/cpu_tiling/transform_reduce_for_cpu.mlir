// RUN: mlir-hlo-opt %s --split-input-file \
// RUN: -xla-cpu-transform-reduce="vector-size=8 tile-size-1d=32 tile-sizes-2d=4,2" \
// RUN: | FileCheck %s

func.func @reduce_add_static(%input: tensor<100x10xf32>,
                        %output: tensor<10xf32>) -> tensor<10xf32> {
  %res = linalg.reduce { arith.addf }
           ins(%input: tensor<100x10xf32>)
           outs(%output: tensor<10xf32>)
           dimensions = [0]
  return %res : tensor<10xf32>
}
// CHECK-LABEL: func @reduce_add_static

//       CHECK:   gml_st.parallel
//       CHECK:     scf.for
//       CHECK:       linalg.reduce
//  CHECK-NEXT:       tensor.insert_slice
//  CHECK-NEXT:       scf.yield
//       CHECK:     gml_st.set_yield

// -----

func.func @reduce_mulf(%input: tensor<?x?xf32>,
                      %output: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %output, %c0 : tensor<?xf32>
  %1 = tensor.empty(%0) : tensor<?xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<?xf32>) -> tensor<?xf32>
  %res = linalg.reduce { arith.mulf }
           ins(%input: tensor<?x?xf32>)
           outs(%2: tensor<?xf32>)
           dimensions = [1]

  return %res : tensor<?xf32>
}

// CHECK-LABEL: func @reduce_mulf

// CHECK:   gml_st.parallel
// CHECK:     linalg.fill
// CHECK:     scf.for
// CHECK:       scf.yield
// CHECK:     scf.for
// CHECK:       scf.for
// CHECK:         scf.yield
// CHECK:       scf.yield
// CHECK:     gml_st.set_yield

// CHECK:   gml_st.parallel
// CHECK:     linalg.fill
// CHECK:     scf.for
// CHECK:       scf.yield
// CHECK:     gml_st.set_yield

// -----

func.func @reduce_map_fuse(%arg0: tensor<10x100xf32>,
    %arg1: tensor<10x100xf32>, %output: tensor<10xf32>) -> tensor<10xf32> {
  %map_init = tensor.empty() : tensor<10x100xf32>
  %mapped = linalg.map { arith.addf }
              ins(%arg0, %arg1 : tensor<10x100xf32>, tensor<10x100xf32>)
              outs(%map_init : tensor<10x100xf32>)
  %res = linalg.reduce { arith.addf }
           ins(%mapped: tensor<10x100xf32>)
           outs(%output: tensor<10xf32>)
           dimensions = [1]
  return %res : tensor<10xf32>
}

// CHECK-LABEL: func @reduce_map_fuse

// CHECK: gml_st.parallel
// CHECK:   scf.for
// CHECK:     linalg.map
// CHECK:     linalg.reduce
// CHECK:   gml_st.set_yield

// CHECK: gml_st.parallel
// CHECK:   scf.for
// CHECK:     linalg.map
// CHECK:     linalg.reduce
// CHECK:   gml_st.set_yield

// -----

func.func @reduce_1d_static(%arg0: tensor<100xf32>) -> tensor<f32> {
  %1 = tensor.empty() : tensor<f32>
  %cst = arith.constant 0.0 : f32
  %init = linalg.fill ins(%cst : f32) outs(%1 : tensor<f32>) -> tensor<f32>
  %res = linalg.reduce { arith.addf }
           ins(%arg0: tensor<100xf32>)
           outs(%init: tensor<f32>)
           dimensions = [0]
  return %res : tensor<f32>
}

// CHECK-LABEL: func @reduce_1d_static(
//  CHECK-SAME: %[[ARG0:.*]]: tensor<100xf32>
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0
//   CHECK-DAG: %[[C4:.*]] = arith.constant 4
//   CHECK-DAG: %[[C32:.*]] = arith.constant 32
//   CHECK-DAG: %[[C96:.*]] = arith.constant 96
//   CHECK-DAG: %[[C100:.*]] = arith.constant 100
//   CHECK-DAG: %[[CST:.*]] = arith.constant 0.0000

//   CHECK-DAG: %[[EMP0:.*]] = tensor.empty() : tensor<f32>
//   CHECK-DAG: %[[EMP1:.*]] = tensor.empty() : tensor<8xf32>
//   CHECK-DAG: %[[FILL0:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMP0]] : tensor<f32>)
//   CHECK-DAG: %[[FILL1:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMP1]] : tensor<8xf32>)

//       CHECK: %[[TILE_RESULT:.*]] = scf.for %[[I:.*]] = %[[C0]] to
//  CHECK-SAME:     %[[C96]] step %[[C32]] iter_args(%[[ACC:.*]] = %[[FILL1]]
//       CHECK:   %[[INPUT_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[I]]] [32] [1]
//       CHECK:   %[[SHAPED_SLICE:.*]] = tensor.expand_shape %[[INPUT_SLICE]]
//       CHECK:   %[[TILED_REDUCE:.*]] = linalg.reduce
//  CHECK-SAME:     ins(%[[SHAPED_SLICE]]
//  CHECK-SAME:     outs(%[[ACC]]
//  CHECK-SAME:     dimensions = [0]
//       CHECK:   scf.yield %[[TILED_REDUCE]]

//       CHECK: %[[HORIZONTAL_REDUCE:.*]] = linalg.reduce
//  CHECK-SAME:   ins(%[[TILE_RESULT]]
//  CHECK-SAME:   outs(%[[FILL0]]
//  CHECK-SAME:   dimensions = [0]

//       CHECK: %[[REMAINDER_RESULT:.*]] = scf.for %[[J:.*]] = %[[C96]] to
//  CHECK-SAME:     %[[C100]] step %[[C32]] iter_args(%[[ACC1:.*]] = %[[HORIZONTAL_REDUCE]]
//       CHECK:   %[[INPUT_SLICE1:.*]] = tensor.extract_slice %[[ARG0]][%[[J]]] [%[[C4]]] [1]
//       CHECK:   %[[REMAINDER_REDUCE:.*]] = linalg.reduce
//  CHECK-SAME:     ins(%[[INPUT_SLICE1]]
//  CHECK-SAME:     outs(%[[ACC1]]
//  CHECK-SAME:     dimensions = [0]
//       CHECK:   scf.yield %[[REMAINDER_REDUCE]]
//       CHECK: return %[[REMAINDER_RESULT]]

// -----

func.func @reduce_1d_dynamic(%arg0: tensor<?xf32>) -> tensor<f32> {
  %1 = tensor.empty() : tensor<f32>
  %cst = arith.constant 0.0 : f32
  %init = linalg.fill ins(%cst : f32) outs(%1 : tensor<f32>) -> tensor<f32>
  %res = linalg.reduce { arith.addf }
           ins(%arg0: tensor<?xf32>)
           outs(%init: tensor<f32>)
           dimensions = [0]
  return %res : tensor<f32>
}

//       CHECK: #map = affine_map<()[s0] -> (s0 - s0 mod 32)>
//       CHECK: #map1 = affine_map<()[s0, s1] -> (s1 - s0)>
// CHECK-LABEL: func @reduce_1d_dynamic(
//  CHECK-SAME: %[[ARG0:.*]]: tensor<?xf32>
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0
//   CHECK-DAG: %[[C32:.*]] = arith.constant 32
//   CHECK-DAG: %[[CST:.*]] = arith.constant 0.0000

//   CHECK-DAG: %[[INPUT_SIZE:.*]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG: %[[TILABLE_BOUND:.*]] = affine.apply #map()[%[[INPUT_SIZE]]]
//   CHECK-DAG: %[[REMAINDER_SIZE:.*]] = affine.apply #map1()[%[[TILABLE_BOUND]], %[[INPUT_SIZE]]]

//       CHECK: %[[TILE_RESULT:.*]] = scf.for %[[I:.*]] = %[[C0]] to
//  CHECK-SAME:     %[[TILABLE_BOUND]] step %[[C32]]
//       CHECK:   %[[TILED_REDUCE:.*]] = linalg.reduce
//       CHECK:   __perfectly_tiled_loop_label__

//      CHECK: %[[HORIZONTAL_REDUCE:.*]] = linalg.reduce

//      CHECK: %[[REMAINDER_RESULT:.*]] = scf.for %[[J:.*]] = %[[TILABLE_BOUND]] to
// CHECK-SAME:     %[[INPUT_SIZE]] step %[[C32]] iter_args(%[[ACC1:.*]] = %[[HORIZONTAL_REDUCE]]
//      CHECK:   %[[INPUT_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[J]]] [%[[REMAINDER_SIZE]]] [1]

//      CHECK: return %[[REMAINDER_RESULT]]

// -----

func.func @reduce_map_fuse_map(%arg0: tensor<10x100xf32>,
    %arg1: tensor<10x100xf32>, %output: tensor<10xf32>) -> tensor<10xf32> {
  %map_init = tensor.empty() : tensor<10x100xf32>
  %reduce_init = tensor.empty() : tensor<10xf32>
  %mapped = linalg.map { arith.addf }
              ins(%arg0, %arg1 : tensor<10x100xf32>, tensor<10x100xf32>)
              outs(%map_init : tensor<10x100xf32>)

  %reduce = linalg.reduce { arith.addf }
              ins(%mapped: tensor<10x100xf32>)
              outs(%reduce_init: tensor<10xf32>)
              dimensions = [1]

  %res = linalg.map { math.absf }
           ins(%reduce: tensor<10xf32>)
           outs(%output : tensor<10xf32>)
  return %res : tensor<10xf32>
}

// CHECK-LABEL:    func @reduce_map_fuse_map

// CHECK: gml_st.parallel
// CHECK:   scf.for
// CHECK:     linalg.map
// CHECK:     linalg.reduce
// CHECK:     scf.yield
// CHECK:   linalg.map
// CHECK:   gml_st.set_yield

// CHECK: gml_st.parallel
// CHECK:   scf.for
// CHECK:     linalg.map
// CHECK:     linalg.reduce
// CHECK:     scf.yield
// CHECK:   linalg.map
// CHECK:   gml_st.set_yield
