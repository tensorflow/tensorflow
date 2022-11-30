// RUN: mlir-hlo-opt %s \
// RUN:     --gml-tiling-cwise="tile-sizes=512,1024 distribute=true" \
// RUN:     --gml-tiling-cwise="tile-sizes=64,128 distribute=true" \
// RUN:     --gml-tiling-cwise="tile-sizes=1,32 distribute=true" --cse | \
// RUN: FileCheck %s

#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @cwise_expr
// CHECK-SAME:  %[[A:.*]]: tensor<?x1024x1024xf32>, %[[B:.*]]: tensor<?x1024x1024xf32>, %[[C:.*]]: tensor<?x1024x1024xf32>
func.func @cwise_expr(%a: tensor<?x1024x1024xf32>, %b: tensor<?x1024x1024xf32>,
    %c: tensor<?x1024x1024xf32>) -> tensor<?x1024x1024xf32> {
  // CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : index
  // CHECK-DAG:   %[[C64:.*]] = arith.constant 64 : index
  // CHECK-DAG:   %[[C128:.*]] = arith.constant 128 : index
  // CHECK-DAG:   %[[C512:.*]] = arith.constant 512 : index
  // CHECK-DAG:   %[[C1024:.*]] = arith.constant 1024 : index
  // CHECK-DAG:   %[[A_D0:.*]] = tensor.dim %[[A]], %[[C0]]
  // CHECK-DAG:   %[[INIT:.*]] = tensor.empty(%[[A_D0]])
  // CHECK:       %[[PLOOP:.*]] = gml_st.parallel
  // CHECK-SAME:      (%[[I:.*]], %[[J:.*]], %[[K:.*]]) = (%[[C0]], %[[C0]], %[[C0]])
  // CHECK-SAME:      to (%[[A_D0]], %[[C1024]], %[[C1024]])
  // CHECK-SAME:      step (%[[C1]], %[[C512]], %[[C1024]])
  // CHECK-DAG:     %[[A_SUB:.*]] = gml_st.materialize %[[A]][%{{.*}}]
  // CHECK-DAG:     %[[B_SUB:.*]] = gml_st.materialize %[[B]][%{{.*}}]
  // CHECK-DAG:     %[[C_SUB:.*]] = gml_st.materialize %[[C]][%{{.*}}]
  // CHECK-DAG:     %[[INIT_SUB:.*]] = gml_st.materialize %[[INIT]][%{{.*}}]
  // CHECK:         %[[PLOOP_:.*]] = gml_st.parallel
  // CHECK-SAME:        (%[[I_:.*]], %[[J_:.*]], %[[K_:.*]]) = (%[[C0]], %[[C0]], %[[C0]])
  // CHECK-SAME:        to (%[[C1]], %[[C512]], %[[C1024]])
  // CHECK-SAME:        step (%[[C1]], %[[C64]], %[[C128]])
  // CHECK-DAG:       %[[A_SUB_SUB:.*]] = gml_st.materialize %[[A_SUB]][%{{.*}}]
  // CHECK-DAG:       %[[B_SUB_SUB:.*]] = gml_st.materialize %[[B_SUB]][%{{.*}}]
  // CHECK-DAG:       %[[C_SUB_SUB:.*]] = gml_st.materialize %[[C_SUB]][%{{.*}}]
  // CHECK-DAG:       %[[INIT_SUB_SUB:.*]] = gml_st.materialize %[[INIT_SUB]][%{{.*}}]
  // CHECK:           %[[PLOOP__:.*]] = gml_st.parallel
  // CHECK-SAME:          (%[[I__:.*]], %[[J__:.*]], %[[K__:.*]]) = (%[[C0]], %[[C0]], %[[C0]])
  // CHECK-SAME:          to (%[[C1]], %[[C64]], %[[C128]])
  // CHECK-SAME:          step (%[[C1]], %[[C1]], %[[C32]])
  // CHECK-DAG:         %[[A_SUB_SUB_SUB:.*]] = gml_st.materialize %[[A_SUB_SUB]][%{{.*}}]
  // CHECK-DAG:         %[[B_SUB_SUB_SUB:.*]] = gml_st.materialize %[[B_SUB_SUB]][%{{.*}}]
  // CHECK-DAG:         %[[INIT_SUB_SUB_SUB:.*]] = gml_st.materialize %[[INIT_SUB_SUB]][%{{.*}}]
  // CHECK:             %[[AB_SUB_SUB_SUB:.*]] = linalg.generic
  // CHECK-SAME:            ins(%[[A_SUB_SUB_SUB]], %[[B_SUB_SUB_SUB]] : tensor<1x1x32xf32>, tensor<1x1x32xf32>)
  // CHECK-SAME:            outs(%[[INIT_SUB_SUB_SUB]] : tensor<1x1x32xf32>)
  // CHECK-DAG:         %[[C_SUB_SUB_SUB:.*]] = gml_st.materialize %[[C_SUB_SUB]][%{{.*}}]
  // CHECK:             %[[ABC_SUB_SUB_SUB:.*]] = linalg.generic
  // CHECK-SAME:            ins(%[[AB_SUB_SUB_SUB]], %[[C_SUB_SUB_SUB]] : tensor<1x1x32xf32>, tensor<1x1x32xf32>)
  // CHECK-SAME:            outs(%[[INIT_SUB_SUB_SUB]] : tensor<1x1x32xf32>)
  // CHECK:             gml_st.set_yield %[[ABC_SUB_SUB_SUB]] into %[[INIT_SUB_SUB]][%{{.*}}]
  // CHECK:           gml_st.set_yield %[[PLOOP__]] into %[[INIT_SUB]][%{{.*}}]
  // CHECK:         gml_st.set_yield %[[PLOOP_]] into %[[INIT]][%{{.*}}]
  // CHECK:       return %[[PLOOP]]
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
