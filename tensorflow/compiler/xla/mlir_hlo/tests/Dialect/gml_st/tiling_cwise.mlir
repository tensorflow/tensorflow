// RUN: mlir-hlo-opt %s \
// RUN:     --gml-tiling-cwise="tile-sizes=4,8 distribute=true distribution-label=test" \
// RUN:     --cse | \
// RUN: FileCheck %s

#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @cwise_expr(%a: tensor<?x1024x1024xf32>, %b: tensor<?x1024x1024xf32>,
    %c: tensor<?x1024x1024xf32>) -> tensor<?x1024x1024xf32> {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %a, %c0 : tensor<?x1024x1024xf32>
  %init = tensor.empty(%d0) : tensor<?x1024x1024xf32>
  %ab = linalg.generic {
      indexing_maps = [#id, #id, #id],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%a, %b : tensor<?x1024x1024xf32>, tensor<?x1024x1024xf32>)
      outs(%init : tensor<?x1024x1024xf32>) {
  ^bb0(%a_: f32, %b_: f32, %_: f32):
    %ab_ = arith.addf %a_, %b_ : f32
    linalg.yield %ab_ : f32
  } -> tensor<?x1024x1024xf32>
  %abc = linalg.generic {
      indexing_maps = [#id, #id, #id],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%ab, %c : tensor<?x1024x1024xf32>, tensor<?x1024x1024xf32>)
      outs(%init : tensor<?x1024x1024xf32>) {
  ^bb0(%ab_: f32, %c_: f32, %_: f32):
    %abc_ = arith.addf %ab_, %c_ : f32
    linalg.yield %abc_ : f32
  } -> tensor<?x1024x1024xf32>
  func.return %abc : tensor<?x1024x1024xf32>
}


// CHECK-LABEL: @cwise_expr
// CHECK-SAME:  %[[A:.*]]: tensor<?x1024x1024xf32>, %[[B:.*]]: tensor<?x1024x1024xf32>, %[[C:.*]]: tensor<?x1024x1024xf32>

// CHECK-DAG:   %[[C1:.*]] = arith.constant 1
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0
// CHECK-DAG:   %[[C1024:.*]] = arith.constant 1024
// CHECK-DAG:   %[[A_D0:.*]] = tensor.dim %[[A]], %[[C0]]
// CHECK-DAG:   %[[INIT:.*]] = tensor.empty(%[[A_D0]])
// CHECK:       %[[ABC:.*]] = gml_st.parallel 
// CHECK-SAME:      (%[[I:.*]], %[[J:.*]], %[[K:.*]]) = (%[[C0]], %[[C0]], %[[C0]])
// CHECK-SAME:      to (%[[A_D0]], %[[C1024]], %[[C1024]])
// CHECK-SAME:      step (%[[C1]], %[[C4]], %[[C8]])
// CHECK-SAME:      distribution ("test")
// CHECK-DAG:     %[[TILE:.*]] = gml_st.tile [%[[I]], %[[J]], %[[K]]] [1, 4, 8] [1, 1, 1]
// CHECK-DAG:     %[[A_SUB:.*]] = gml_st.materialize %[[A]][%[[TILE]]]
// CHECK-DAG:     %[[B_SUB:.*]] = gml_st.materialize %[[B]][%[[TILE]]]
// CHECK-DAG:     %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]][%[[TILE]]]
// CHECK-DAG:     %[[AB_SUB:.*]] = linalg.generic 
// CHECK-SAME:        ins(%[[A_SUB]], %[[B_SUB]] : tensor<1x4x8xf32>, tensor<1x4x8xf32>) 
// CHECK-SAME:        outs(%[[INIT_SUB]] : tensor<1x4x8xf32>)
// CHECK-DAG:     %[[C_SUB:.*]] = gml_st.materialize %[[C]][%[[TILE]]]
// CHECK-DAG:     %[[ABC_SUB:.*]] = linalg.generic
// CHECK-SAME:        ins(%[[AB_SUB]], %[[C_SUB]] : tensor<1x4x8xf32>, tensor<1x4x8xf32>) 
// CHECK-SAME:        outs(%[[INIT_SUB]] : tensor<1x4x8xf32>)
// CHECK:         gml_st.set_yield %[[ABC_SUB]] into %[[INIT]][%[[TILE]]]
// CHECK:       return %[[ABC]]
