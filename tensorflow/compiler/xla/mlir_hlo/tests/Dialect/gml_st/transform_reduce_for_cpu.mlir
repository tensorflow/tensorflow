// RUN: mlir-hlo-opt %s -xla-cpu-transform-reduce="tile-sizes=2" --split-input-file | FileCheck %s

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

//       CHECK:         %[[REDUCED:.*]] = linalg.reduce
//  CHECK-NEXT:           ins(%[[IN:.*]] : tensor<100x2xf32>)
//  CHECK-NEXT:           outs(%[[OUT:.*]] : tensor<2xf32>)
//  CHECK-NEXT:           dimensions = [0]

//       CHECK:       gml_st.set_yield %[[REDUCED]]

// -----

func.func @reduce_mulf(%input: tensor<?x?xf32>,
                      %output: tensor<?xf32>) -> tensor<?xf32> {
  %res = linalg.reduce ins(%input: tensor<?x?xf32>)
                     outs(%output: tensor<?xf32>)
                     dimensions = [1]
          (%in: f32, %init: f32) {
            %0 = arith.mulf %in, %init : f32
            linalg.yield %0 : f32
          }
  return %res : tensor<?xf32>
}

// CHECK-LABEL:     func @reduce_mulf(
//  CHECK-SAME:       %[[IN:.*]]: tensor<?x?xf32>,
//  CHECK-SAME:       %[[OUT:.*]]: tensor<?xf32>)
//  CHECK-SAME:       -> tensor<?xf32> {

//   CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:       %[[DIM0:.*]] = tensor.dim %[[IN]], %[[C0]]

//       CHECK:       gml_st.parallel (%[[I:.*]]) = (%[[C0]])

//       CHECK:         %[[REDUCED:.*]] = linalg.reduce
//  CHECK-NEXT:           ins(%[[IN:.*]] : tensor<?x?xf32>)
//  CHECK-NEXT:           outs(%[[OUT:.*]] : tensor<?xf32>)
//  CHECK-NEXT:           dimensions = [1]

//       CHECK:       gml_st.set_yield %[[REDUCED]]