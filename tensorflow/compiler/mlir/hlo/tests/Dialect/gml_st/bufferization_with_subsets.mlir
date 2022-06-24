// RUN: mlir-hlo-opt %s -test-gml-st-bufferization -canonicalize -cse \
// RUN:   -split-input-file | FileCheck %s

func.func @subset_space(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %dim_0 = tensor.dim %input, %c0 : tensor<?x?xf32>
  %dim_1 = tensor.dim %input, %c1 : tensor<?x?xf32>

  %space = gml_st.space [%dim_0, %dim_1] : !gml_st.tile<?x?>
  %identity = gml_st.materialize %input[%space]
    : tensor<?x?xf32>[!gml_st.tile<?x?>]

  return %identity : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @subset_space(
// CHECK-SAME:    %[[ARG:.*]]: memref<?x?xf32>)
// CHECK-NEXT:  return %[[ARG]] : memref<?x?xf32>

// -----

func.func @subset_tile(%input: tensor<?x?xf32>) -> tensor<2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %dim_0 = tensor.dim %input, %c0 : tensor<?x?xf32>
  %dim_1 = tensor.dim %input, %c1 : tensor<?x?xf32>

  %space = gml_st.space [%dim_0, %dim_1] : !gml_st.tile<?x?>
  %tile = gml_st.tile %space[0, 1][2, 4][1, 1]
    : !gml_st.tile<?x?> to !gml_st.tile<2x4>

  %slice = gml_st.materialize %input[%tile]
    : tensor<?x?xf32>[!gml_st.tile<2x4>]

  return %slice : tensor<2x4xf32>
}
// CHECK-LABEL: func.func @subset_tile(
// CHECK-SAME:    %[[ARG:.*]]: memref<?x?xf32>)
// CHECK-NEXT:  %[[VIEW:.*]] = memref.subview %[[ARG]][0, 1] [2, 4] [0, 1]
// CHECK-NEXT:  %[[ALLOC:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK-NEXT:  memref.copy %[[VIEW]], %[[ALLOC]]
// CHECK-NEXT:  return %[[ALLOC]] : memref<2x4xf32>

// -----

func.func @subset_point(%input: tensor<?x?xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %dim_0 = tensor.dim %input, %c0 : tensor<?x?xf32>
  %dim_1 = tensor.dim %input, %c1 : tensor<?x?xf32>

  %space = gml_st.space [%dim_0, %dim_1] : !gml_st.tile<?x?>
  %tile = gml_st.tile %space[0, 1][2, 4][1, 1]
    : !gml_st.tile<?x?> to !gml_st.tile<2x4>
  %pt = gml_st.point %tile[0, 1] : !gml_st.tile<2x4> to !gml_st.point

  %element = gml_st.materialize %input[%pt]
    : tensor<?x?xf32>[!gml_st.point]

  return %element : f32
}
// CHECK-LABEL: func.func @subset_point(
// CHECK-SAME:    %[[ARG:.*]]: memref<?x?xf32>)
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:  %[[VIEW:.*]] = memref.subview %[[ARG]][0, 1] [2, 4] [0, 1]
// CHECK-NEXT:  %[[ELEM:.*]] = memref.load %[[VIEW]][%[[C0]], %[[C1]]]
// CHECK-NEXT:  return %[[ELEM]] : f32
