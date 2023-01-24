// RUN: mlir-hlo-opt %s \
// RUN: -xla-cpu-transform-reduce="vector-size=8 tile-size-1d=32 tile-sizes-2d=4,2" \
// RUN: --split-input-file \
// RUN: | FileCheck %s --check-prefixes=CHECK,PEELED
// RUN: mlir-hlo-opt %s \
// RUN: -xla-cpu-transform-reduce="vector-size=8 tile-size-1d=32 tile-sizes-2d=4,2" \
// RUN: --split-input-file \
// RUN: | FileCheck %s --check-prefixes=MARKED

func.func @reduce_add_static(%input: tensor<100x10xf32>,
                        %output: tensor<10xf32>) -> tensor<10xf32> {
  %res = linalg.reduce { arith.addf }
           ins(%input: tensor<100x10xf32>)
           outs(%output: tensor<10xf32>)
           dimensions = [0]
  return %res : tensor<10xf32>
}

// CHECK-LABEL:     func @reduce_add_static(
//  CHECK-SAME:       %[[IN:.*]]: tensor<100x10xf32>,
//  CHECK-SAME:       %[[OUT:.*]]: tensor<10xf32>)
//  CHECK-SAME:       -> tensor<10xf32> {

//       CHECK:       %[[C0:.*]] = arith.constant 0 : index

//       CHECK:       gml_st.parallel (%[[I:.*]]) = (%[[C0]])
//       CHECK:         %[[IN_SLICE_1:.*]] = tensor.extract_slice %[[IN]]
//       CHECK:         %[[OUT_SLICE_1:.*]] = tensor.extract_slice %[[OUT]]

//       CHECK:         %[[FOR:.*]] = gml_st.for (%[[J:.*]]) = (%[[C0]])
//       CHECK:           %[[IN_SLICE_2:.*]] = tensor.extract_slice
//       CHECK:           %[[OUT_SLICE_2:.*]] = tensor.extract_slice

//       CHECK:           %[[REDUCED:.*]] = linalg.reduce
//  CHECK-SAME:             ins(%[[IN_SLICE_2]] : tensor<2x?xf32>)
//  CHECK-SAME:             outs(%[[OUT_SLICE_2]] : tensor<?xf32>)
//  CHECK-SAME:             dimensions = [0]

//       CHECK:           gml_st.set_yield %[[REDUCED]]
//       CHECK:         gml_st.set_yield %[[FOR]]

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

// PEELED-LABEL:     func @reduce_mulf(
//  PEELED-SAME:       %[[IN:.*]]: tensor<?x?xf32>,
//  PEELED-SAME:       %[[OUT:.*]]: tensor<?xf32>)
//  PEELED-SAME:       -> tensor<?xf32> {

//   PEELED-DAG:       %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//   PEELED-DAG:       %[[C0:.*]] = arith.constant 0 : index
//   PEELED-DAG:       %[[C1:.*]] = arith.constant 1 : index
//       PEELED:       %[[INIT:.*]] = tensor.empty
//       PEELED:       %[[DIM0:.*]] = tensor.dim %[[IN]], %[[C0]]

//       PEELED:       %[[MAIN_PAR:.*]] = gml_st.parallel (%[[I:.*]]) = (%[[C0]]) to (%[[IUB:.*]]) step
//       PEELED:         %[[MAIN_SLICE:.*]] = tensor.extract_slice %[[INIT]]
//       PEELED:         %[[MAIN_FILL:.*]] = linalg.fill{{.*}}outs(%[[MAIN_SLICE]]
//       PEELED:         %[[MAIN_FOR:.*]] = gml_st.for (%[[J:.*]]) = (%[[C0]]) to (%[[JUB:.*]]) {{.*}} outs ({{.*}} = %[[MAIN_FILL]]:
//       PEELED:           %[[MAIN_PAR_MAIN_FOR_REDUCE:.*]] = linalg.reduce
//       PEELED:           gml_st.set_yield %[[MAIN_PAR_MAIN_FOR_REDUCE]]
//       PEELED:         %[[REM_FOR:.*]] = gml_st.for (%[[J:.*]]) = (%[[JUB]]) {{.*}} outs (%[[REM_FOR_ARG:.*]] = %[[MAIN_FOR]]:
//       PEELED:           %[[REM_FOR_SLICE:.*]] = tensor.extract_slice %[[REM_FOR_ARG]]
//       PEELED:           %[[SCALAR_REM_FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]]) {{.*}} outs ({{.*}} = %[[REM_FOR_SLICE]]:
//       PEELED:             %[[MAIN_PAR_REM_FOR_REDUCE:.*]] = linalg.reduce
//       PEELED:             gml_st.set_yield %[[MAIN_PAR_REM_FOR_REDUCE]]
//       PEELED:           gml_st.set_yield %[[SCALAR_REM_FOR]]
//       PEELED:         gml_st.set_yield %[[REM_FOR]]

//       PEELED:       %[[REM_PAR:.*]] = gml_st.parallel (%[[I:.*]]) = (%[[IUB]])
//       PEELED:         %[[REM_SLICE:.*]] = tensor.extract_slice %[[MAIN_PAR]]
//       PEELED:         %[[REM_FILL:.*]] = linalg.fill{{.*}}outs(%[[REM_SLICE]]
//       PEELED:         %[[REM_FOR:.*]] = gml_st.for (%[[J:.*]]) = (%[[C0]]) {{.*}} outs ({{.*}} = %[[REM_FILL]]:
//       PEELED:           %[[REM_PAR_REDUCE:.*]] = linalg.reduce
//       PEELED:           gml_st.set_yield %[[REM_PAR_REDUCE]]
//       PEELED:         gml_st.set_yield %[[REM_FOR]]

// -----

// MARKED-LABEL:     func @reduce_mulf(
//       MARKED:       %[[C0:.*]] = arith.constant 0 : index
//       MARKED:       gml_st.parallel (%[[I:.*]]) = (%[[C0]]) to (%[[IUB:.*]]) step
//       MARKED:         gml_st.for (%[[J:.*]]) = (%[[C0]]) to (%[[JUB:.*]]) step
//       MARKED:         __perfectly_tiled_loop_label__
//       MARKED:         gml_st.for (%[[J:.*]]) = (%[[JUB]])
//       MARKED:         } {__peeling_applied_label__}
//       MARKED:       } {__peeling_applied_label__}

//       MARKED:       gml_st.parallel (%[[I:.*]]) = (%[[IUB]])
//       MARKED:         gml_st.for (%[[J:.*]]) = (%[[C0]])
//       MARKED:         } {__peeling_applied_label__}
//       MARKED:       } {__peeling_applied_label__}

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

// CHECK-LABEL:     func @reduce_map_fuse(
//  CHECK-SAME:       %[[ARG0:[0-9a-zA-Z]*]]: tensor<10x100xf32>,
//  CHECK-SAME:       %[[ARG1:.*]]: tensor<10x100xf32>,
//  CHECK-SAME:       %[[OUT:.*]]: tensor<10xf32>)
//  CHECK-SAME:       -> tensor<10xf32> {

//   CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:       %[[INIT:.*]] = tensor.empty

//       CHECK:       gml_st.parallel (%[[I:.*]]) = (%[[C0]])
//       CHECK:         %[[ARG0_SLICE_1:.*]] = tensor.extract_slice %[[ARG0]]
//       CHECK:         %[[ARG1_SLICE_1:.*]] = tensor.extract_slice %[[ARG1]]
//       CHECK:         %[[INIT_SLICE_1:.*]] = tensor.extract_slice %[[INIT]]
//       CHECK:         %[[OUT_SLICE_1:.*]] = tensor.extract_slice %[[OUT]]

//       CHECK:         %[[FOR:.*]] = gml_st.for (%[[J:.*]]) = (%[[C0]])
//       CHECK:           %[[ARG0_SLICE_2:.*]] = tensor.extract_slice
//       CHECK:           %[[ARG1_SLICE_2:.*]] = tensor.extract_slice
//       CHECK:           %[[INIT_SLICE_2:.*]] = tensor.extract_slice
//       CHECK:           %[[MAPPED:.*]] = linalg.map
//  CHECK-SAME:             ins(%[[ARG0_SLICE_2]], %[[ARG1_SLICE_2]]
//  CHECK-SAME:             outs(%[[INIT_SLICE_2]] : tensor<?x2xf32>)

//       CHECK:           %[[OUT_SLICE_2:.*]] = tensor.extract_slice
//       CHECK:           %[[REDUCED:.*]] = linalg.reduce
//  CHECK-SAME:             ins(%[[MAPPED]] : tensor<?x2xf32>)
//  CHECK-SAME:             outs(%[[OUT_SLICE_2]] : tensor<?xf32>)
//  CHECK-SAME:             dimensions = [1]

//       CHECK:           gml_st.set_yield %[[REDUCED]]
//       CHECK:         gml_st.set_yield %[[FOR]]

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

//       CHECK: %[[TILE_RESULT:.*]] = gml_st.for (%[[I:.*]]) = (%[[C0]]) to
//  CHECK-SAME:     (%[[C96]]) step (%[[C32]]) outs (%[[ACC:.*]] = %[[FILL1]]
//       CHECK:   %[[INPUT_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[I]]] [32] [1]
//       CHECK:   %[[SHAPED_SLICE:.*]] = tensor.expand_shape %[[INPUT_SLICE]]
//       CHECK:   %[[TILED_REDUCE:.*]] = linalg.reduce
//  CHECK-SAME:     ins(%[[SHAPED_SLICE]]
//  CHECK-SAME:     outs(%[[ACC]]
//  CHECK-SAME:     dimensions = [0]
//       CHECK:   gml_st.set_yield %[[TILED_REDUCE]]
//  CHECK-NEXT: } : tensor<8xf32>

//       CHECK: %[[HORIZONTAL_REDUCE:.*]] = linalg.reduce
//  CHECK-SAME:   ins(%[[TILE_RESULT]]
//  CHECK-SAME:   outs(%[[FILL0]]
//  CHECK-SAME:   dimensions = [0]

//       CHECK: %[[REMAINDER_RESULT:.*]] = gml_st.for (%[[J:.*]]) = (%[[C96]]) to
//  CHECK-SAME:     (%[[C100]]) step (%[[C32]]) outs (%[[ACC1:.*]] = %[[HORIZONTAL_REDUCE]]
//       CHECK:   %[[INPUT_SLICE1:.*]] = tensor.extract_slice %[[ARG0]][%[[J]]] [%[[C4]]] [1]
//       CHECK:   %[[REMAINDER_REDUCE:.*]] = linalg.reduce
//  CHECK-SAME:     ins(%[[INPUT_SLICE1]]
//  CHECK-SAME:     outs(%[[ACC1]]
//  CHECK-SAME:     dimensions = [0]
//       CHECK:   gml_st.set_yield %[[REMAINDER_REDUCE]]
//  CHECK-NEXT: } : tensor<f32>
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

//       CHECK: %[[TILE_RESULT:.*]] = gml_st.for (%[[I:.*]]) = (%[[C0]]) to
//  CHECK-SAME:     (%[[TILABLE_BOUND]]) step (%[[C32]])
//       CHECK:   %[[TILED_REDUCE:.*]] = linalg.reduce
//       CHECK:   __perfectly_tiled_loop_label__

//      CHECK: %[[HORIZONTAL_REDUCE:.*]] = linalg.reduce

//      CHECK: %[[REMAINDER_RESULT:.*]] = gml_st.for (%[[J:.*]]) = (%[[TILABLE_BOUND]]) to
// CHECK-SAME:     (%[[INPUT_SIZE]]) step (%[[C32]]) outs (%[[ACC1:.*]] = %[[HORIZONTAL_REDUCE]]
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

// CHECK-LABEL:    func @reduce_map_fuse_map(
// CHECK:      %[[C0:.*]] = arith.constant 0 : index

// CHECK:      gml_st.parallel (%[[I:.*]]) = (%[[C0]])
// CHECK:        gml_st.for
// CHECK:          %[[MAP:.*]] = linalg.map
// CHECK:          %[[REDUCE:.*]] = linalg.reduce
// CHECK:          gml_st.set_yield %[[REDUCE]]

// CHECK:        %[[MAP:.*]] = linalg.map
// CHECK:        gml_st.set_yield %[[MAP]]

// CHECK:      gml_st.parallel
// CHECK:        gml_st.for
// CHECK:          %[[MAP:.*]] = linalg.map
// CHECK:          %[[REDUCE:.*]] = linalg.reduce
// CHECK:          gml_st.set_yield %[[REDUCE]]

// CHECK:        %[[MAP:.*]] = linalg.map
// CHECK:        gml_st.set_yield %[[MAP]]
