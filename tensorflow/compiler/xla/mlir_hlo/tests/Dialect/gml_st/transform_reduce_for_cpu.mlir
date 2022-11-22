// RUN: mlir-hlo-opt %s -xla-cpu-transform-reduce="tile-sizes=4,2" --split-input-file | FileCheck %s

func.func @reduce_add_static(%input: tensor<100x10xf32>,
                        %output: tensor<10xf32>) -> tensor<10xf32> {
  %res = linalg.reduce ins(%input: tensor<100x10xf32>)
                     outs(%output: tensor<10xf32>)
                     dimensions = [0]
          (%in: f32, %init: f32) {
            %0 = arith.addf %in, %init : f32
            linalg.yield %0 : f32
          }
  return %res : tensor<10xf32>
}

// CHECK-LABEL:     func @reduce_add_static(
//  CHECK-SAME:       %[[IN:.*]]: tensor<100x10xf32>,
//  CHECK-SAME:       %[[OUT:.*]]: tensor<10xf32>)
//  CHECK-SAME:       -> tensor<10xf32> {

//       CHECK:       %[[C0:.*]] = arith.constant 0 : index

//       CHECK:       gml_st.parallel (%[[I:.*]]) = (%[[C0]])
//  CHECK-NEXT:         %[[MIN:.*]] = affine.min

//       CHECK:         %[[OUT_DIM:.*]] = tensor.dim %[[TILED_IN:.*]], %[[C1:.*]]
//       CHECK:         %[[FOR:.*]] = gml_st.for (%[[J:.*]]) = (%[[C0]])

//       CHECK:           %[[REDUCED:.*]] = linalg.reduce
//  CHECK-NEXT:             ins(%[[IN:.*]] : tensor<2x?xf32>)
//  CHECK-NEXT:             outs(%[[OUT:.*]] : tensor<?xf32>)
//  CHECK-NEXT:             dimensions = [0]

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
  %res = linalg.reduce ins(%input: tensor<?x?xf32>)
                     outs(%2: tensor<?xf32>)
                     dimensions = [1]
          (%in: f32, %init: f32) {
            %mulf = arith.mulf %in, %init : f32
            linalg.yield %mulf : f32
          }
  return %res : tensor<?xf32>
}

// CHECK-LABEL:     func @reduce_mulf(
//  CHECK-SAME:       %[[IN:.*]]: tensor<?x?xf32>,
//  CHECK-SAME:       %[[OUT:.*]]: tensor<?xf32>)
//  CHECK-SAME:       -> tensor<?xf32> {

//   CHECK-DAG:       %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
//       CHECK:       %[[DIM0:.*]] = tensor.dim %[[IN]], %[[C0]]

//       CHECK:       gml_st.parallel (%[[I:.*]]) = (%[[C0]])
//  CHECK-NEXT:         %[[MIN1:.*]] = affine.min

//       CHECK:         %[[FILL:.*]] = linalg.fill
//  CHECK-SAME:           ins(%[[CST]] : f32)
//  CHECK-SAME:           outs(%[[TILED_OUT:.*]] : tensor<?xf32>)

//       CHECK:         %[[DIM1:.*]] = tensor.dim %[[TILED_IN:.*]], %[[C0]]
//       CHECK:         %[[DIM2:.*]] = tensor.dim %[[TILED_IN]], %[[C1]]
//       CHECK:         %[[FOR:.*]] = gml_st.for (%[[J:.*]]) = (%[[C0]])
//  CHECK-NEXT:           %[[MIN2:.*]] = affine.min

//       CHECK:           %[[REDUCED:.*]] = linalg.reduce
//  CHECK-NEXT:             ins(%[[IN:.*]] : tensor<?x?xf32>)
//  CHECK-NEXT:             outs(%[[OUT:.*]] : tensor<?xf32>)
//  CHECK-NEXT:             dimensions = [1]

//       CHECK:           gml_st.set_yield %[[REDUCED]]
//       CHECK:         gml_st.set_yield %[[FOR]]