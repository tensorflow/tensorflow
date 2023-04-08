// RUN: mlir-hlo-opt %s --gml-compose-extract-insert-slice --split-input-file \
// RUN: | FileCheck %s

func.func @compose_slices(%arg: tensor<?x?xf32>, %i: index, %j: index,
    %k: index, %n: index, %a: index, %b: index) -> tensor<4x?xf32> {
  %4 = tensor.extract_slice %arg[%i, %j] [4, 128] [2, %a]
    : tensor<?x?xf32> to tensor<4x128xf32>
  %5 = tensor.extract_slice %4[0, %k] [4, %n] [1, %b]
    : tensor<4x128xf32> to tensor<4x?xf32>
  return %5 : tensor<4x?xf32>
}
// CHECK-LABEL: @compose_slices
// CHECK-SAME:  %[[ARG:[a-z0-9]+]]: tensor<?x?xf32>, %[[I:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[J:[a-z0-9]+]]: index, %[[K:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[N:[a-z0-9]+]]: index, %[[A:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[B:[a-z0-9]+]]: index)

// CHECK-DAG:  %[[J_PLUS_AK:.*]] = affine.apply
// CHECK-DAG:  %[[AB:.*]] = affine.apply
// CHECK-NEXT: %[[RES:.*]] = tensor.extract_slice %[[ARG]]
// CHECK-SAME:   [%[[I]], %[[J_PLUS_AK]]] [4, %[[N]]] [2, %[[AB]]]
// CHECK-SAME:   : tensor<?x?xf32>

// -----

func.func @compose_extract_of_slice(%arg: tensor<?x?xf32>, %i: index, %j: index,
    %k: index, %l: index) -> f32 {
  %slice = tensor.extract_slice %arg[%i, %j] [4, 128] [2, %l]
    : tensor<?x?xf32> to tensor<4x128xf32>
  %c1 = arith.constant 1 : index
  %pt = tensor.extract %slice[%c1, %k] : tensor<4x128xf32>
  return %pt : f32
}
// CHECK-DAG: #[[$MAP0:.*]] = affine_map<()[s0] -> (s0 + 2)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>

// CHECK-LABEL: func.func @compose_extract_of_slice
// CHECK-SAME:   (%[[ARG:.*]]: tensor<?x?xf32>,
// CHECK-SAME:    %[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index,
// CHECK-SAME:    %[[L:.*]]: index) -> f32 {

// CHECK:       %[[X:.*]] = affine.apply #[[$MAP0]]()[%[[I]]]
// CHECK:       %[[Y:.*]] = affine.apply #[[$MAP1]]()[%[[K]], %[[L]], %[[J]]]
// CHECK:       tensor.extract %[[ARG]][%[[X]], %[[Y]]] : tensor<?x?xf32>

