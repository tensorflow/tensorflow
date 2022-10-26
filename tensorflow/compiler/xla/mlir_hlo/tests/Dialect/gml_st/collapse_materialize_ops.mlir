// RUN: mlir-hlo-opt %s --gml-collapse-materialize-ops | \
// RUN: FileCheck %s

func.func @compose_tiles(%arg: tensor<?x?xf32>, %i: index, %j: index, %k: index,
    %n: index, %a: index, %b: index) -> tensor<4x?xf32> {
  %1 = gml_st.tile [%i, %j] [4, 128] [2, %a]
      : !gml_st.tile<4x128>
  %4 = gml_st.materialize %arg[%1] : tensor<?x?xf32>[!gml_st.tile<4x128>] to tensor<4x128xf32>
  %3 = gml_st.tile [0, %k] [4, %n] [1, %b]
      : !gml_st.tile<4x?>
  %5 = gml_st.materialize %4[%3] : tensor<4x128xf32>[!gml_st.tile<4x?>] to tensor<4x?xf32>
  return %5 : tensor<4x?xf32>
}
// CHECK-LABEL: @compose_tiles
// CHECK-SAME:  %[[ARG:[a-z0-9]+]]: tensor<?x?xf32>, %[[I:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[J:[a-z0-9]+]]: index, %[[K:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[N:[a-z0-9]+]]: index, %[[A:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[B:[a-z0-9]+]]: index)

// CHECK-DAG:  %[[AK:.*]] = arith.muli %[[A]], %[[K]]
// CHECK-DAG:  %[[J_PLUS_AK:.*]] = arith.addi %[[J]], %[[AK]]
// CHECK-DAG:  %[[AB:.*]] = arith.muli %[[A]], %[[B]]
// CHECK:      %[[TILE:.*]] = gml_st.tile [%[[I]], %[[J_PLUS_AK]]] [4, %[[N]]]
// CHECK-SAME:   [2, %[[AB]]] : !gml_st.tile<4x?>
// CHECK-NEXT:  %[[RES:.*]] = gml_st.materialize %[[ARG]][%[[TILE]]]
// CHECK-SAME:    : tensor<?x?xf32>[!gml_st.tile<4x?>]
