// RUN: mlir-hlo-opt %s --gml-st-optimize-linalg-ops-pass \
// RUN: --split-input-file \
// RUN: | FileCheck %s

func.func @map_no_inputs(%arg: tensor<32xf32>) -> tensor<32xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<32xf32>

  %res = linalg.map
           outs(%init: tensor<32xf32>)
           () {
             linalg.yield %c0 : f32
           }
  func.return %res : tensor<32xf32>
}

// CHECK-LABEL:  @map_no_inputs
// CHECK-DAG:      %[[CST:.*]] = arith.constant
// CHECK-DAG:      %[[INIT:.*]] = tensor.empty
// CHECK:          linalg.fill
// CHECK-SAME:       ins(%[[CST]]
// CHECK-SAME:       outs(%[[INIT]]

// -----

func.func @map_dense_constant_operand(%arg: tensor<32xf32>) -> tensor<32xf32> {
  %c0 = arith.constant dense<0.0> : tensor<32xf32>
  %init = tensor.empty() : tensor<32xf32>

  %res = linalg.map { arith.maxf }
           ins(%arg, %c0: tensor<32xf32>, tensor<32xf32>)
           outs(%init: tensor<32xf32>)
  func.return %res : tensor<32xf32>
}

// CHECK-LABEL:  @map_dense_constant_operand
// CHECK-SAME:       (%[[ARG:.*]]: tensor<32xf32>)
// CHECK-DAG:      %[[CST:.*]] = arith.constant 0.0
// CHECK-DAG:      %[[INIT:.*]] = tensor.empty
// CHECK:          linalg.map
// CHECK-SAME:       ins(%[[ARG]]
// CHECK-SAME:       outs(%[[INIT]]
// CHECK-NEXT:       (%[[BBARG:.*]]: f32)
// CHECK-NEXT:         arith.maxf %[[BBARG]], %[[CST]]

// -----

func.func @map_fill_operand(%arg: tensor<32xf32>) -> tensor<32xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<32xf32>

  %filled = linalg.fill ins(%c0 : f32)
              outs(%init: tensor<32xf32>) -> tensor<32xf32>

  %res = linalg.map { arith.maxf }
           ins(%arg, %filled: tensor<32xf32>, tensor<32xf32>)
           outs(%init: tensor<32xf32>)
  func.return %res : tensor<32xf32>
}

// CHECK-LABEL:  @map_fill_operand
// CHECK-SAME:       (%[[ARG:.*]]: tensor<32xf32>)
// CHECK-DAG:      %[[CST:.*]] = arith.constant 0.0
// CHECK-DAG:      %[[INIT:.*]] = tensor.empty
// CHECK:          linalg.map
// CHECK-SAME:       ins(%[[ARG]]
// CHECK-SAME:       outs(%[[INIT]]
// CHECK-NEXT:       (%[[BBARG:.*]]: f32)
// CHECK-NEXT:         arith.maxf %[[BBARG]], %[[CST]]

// -----

func.func @map_all_constant_operand(%select: i1) -> tensor<32xf32> {
  %c0 = arith.constant dense<0.0> : tensor<32xf32>
  %c1 = arith.constant 1.0 : f32
  %init = tensor.empty() : tensor<32xf32>

  %filled = linalg.fill ins(%c1 : f32)
              outs(%init: tensor<32xf32>) -> tensor<32xf32>

  %res = linalg.map
           ins(%c0, %filled: tensor<32xf32>, tensor<32xf32>)
           outs(%init: tensor<32xf32>)
           (%lhs : f32, %rhs : f32) {
             %0 = arith.select %select, %lhs, %rhs : f32
             linalg.yield %0 : f32
           }
  func.return %res : tensor<32xf32>
}

// CHECK-LABEL:  @map_all_constant_operand
// CHECK-DAG:      %[[C0:.*]] = arith.constant 0.0
// CHECK-DAG:      %[[C1:.*]] = arith.constant 1.0
// CHECK-DAG:      %[[INIT:.*]] = tensor.empty
// CHECK-DAG:      %[[VAL:.*]] = arith.select
// CHECK:          linalg.fill
// CHECK-SAME:       ins(%[[VAL]]
// CHECK-SAME:       outs(%[[INIT]]

// -----

func.func @broadcast_of_splat() -> tensor<32x64xf32> {
  %c0 = arith.constant dense<0.0> : tensor<32xf32>
  %init = tensor.empty() : tensor<32x64xf32>

  %bcast = linalg.broadcast
    ins(%c0: tensor<32xf32>)
    outs(%init: tensor<32x64xf32>)
    dimensions = [1]
  func.return %bcast : tensor<32x64xf32>
}
// CHECK-LABEL:  @broadcast_of_splat
// CHECK-DAG:      %[[CST:.*]] = arith.constant
// CHECK-DAG:      %[[INIT:.*]] = tensor.empty
// CHECK:          linalg.fill
// CHECK-SAME:       ins(%[[CST]]
// CHECK-SAME:       outs(%[[INIT]]

// -----

func.func @broadcast_of_single_element_tensor(%arg: tensor<f32>)
    -> tensor<32xf32> {
  %init = tensor.empty() : tensor<32xf32>
  %bcast = linalg.broadcast
    ins(%arg: tensor<f32>)
    outs(%init: tensor<32xf32>)
    dimensions = [0]
  func.return %bcast : tensor<32xf32>
}
// CHECK-LABEL:  @broadcast_of_single_element_tensor
// CHECK-SAME:       (%[[ARG:.*]]: tensor<f32>)

// CHECK-DAG:      %[[INIT:.*]] = tensor.empty
// CHECK-DAG:      %[[EXTRACT:.*]] = tensor.extract %[[ARG]]
// CHECK:          linalg.fill
// CHECK-SAME:       ins(%[[EXTRACT]]
// CHECK-SAME:       outs(%[[INIT]]
