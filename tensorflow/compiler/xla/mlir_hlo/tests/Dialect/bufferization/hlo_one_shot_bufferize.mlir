// RUN: mlir-hlo-opt %s --hlo-one-shot-bufferize | FileCheck %s

// TODO(frgossen): Move tests upstream.

func.func @id(%arg : tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
  func.return %arg : tensor<1x2x3xf32>
}

// CHECK: @id(%[[ARG0:.*]]: memref<1x2x3xf32>) -> memref<1x2x3xf32>
// CHECK:   return %[[ARG0]]

func.func @id_select(%pred : i1, %arg : tensor<1x2x3xf32>)
    -> tensor<1x2x3xf32> {
  %0 = arith.select %pred, %arg, %arg : tensor<1x2x3xf32>
  func.return %0 : tensor<1x2x3xf32>
}

// CHECK: @id_select(%[[ARG0_0:.*]]: i1, %[[ARG1:.*]]: memref<1x2x3xf32>) -> memref<1x2x3xf32>
// CHECK:   %[[SELECT:.*]] = arith.select %[[ARG0_0]], %[[ARG1]], %[[ARG1]]
// CHECK:   return %[[SELECT]]

func.func @ite(%pred : i1, %lhs : tensor<1x2x3xf32>,
    %rhs : tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
  %0 = scf.if %pred -> tensor<1x2x3xf32> {
    scf.yield %lhs : tensor<1x2x3xf32>
  } else {
    scf.yield %rhs : tensor<1x2x3xf32>
  }
  func.return %0 : tensor<1x2x3xf32>
}

// CHECK: @ite(%[[ARG0_1:.*]]: i1, %[[ARG1_0:.*]]: memref<1x2x3xf32>, %[[ARG2:.*]]: memref<1x2x3xf32>) -> memref<1x2x3xf32>
// CHECK:   %[[IF:.*]] = scf.if %[[ARG0_1]]
// CHECK:     scf.yield %[[ARG1_0]]
// CHECK:   else
// CHECK:     scf.yield %[[ARG2]]
// CHECK:   return %[[IF]]

func.func @ite_select(%pred : i1, %lhs : tensor<1x2x3xf32>,
    %rhs : tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
  %0 = arith.select %pred, %lhs, %rhs : tensor<1x2x3xf32>
  func.return %0 : tensor<1x2x3xf32>
}

// CHECK: @ite_select(%[[ARG0_2:.*]]: i1, %[[ARG1_1:.*]]: memref<1x2x3xf32>, %[[ARG2_0:.*]]: memref<1x2x3xf32>) -> memref<1x2x3xf32>
// CHECK:   %[[SELECT_0:.*]] = arith.select %[[ARG0_2]], %[[ARG1_1]], %[[ARG2_0]]
// CHECK:   return %[[SELECT_0]]

func.func @may_reuse(%pred : i1, %arg : tensor<1x2x3xf32>)
    -> tensor<1x2x3xf32> {
  %0 = scf.if %pred -> tensor<1x2x3xf32> {
    scf.yield %arg : tensor<1x2x3xf32>
  } else {
    %new_tensor = bufferization.alloc_tensor() : tensor<1x2x3xf32>
    scf.yield %new_tensor : tensor<1x2x3xf32>
  }
  func.return %0 : tensor<1x2x3xf32>
}

// CHECK: @may_reuse(%[[ARG0_3:.*]]: i1, %[[ARG1_2:.*]]: memref<1x2x3xf32>) -> memref<1x2x3xf32>
// CHECK:   %[[IF_0:.*]] = scf.if %[[ARG0_3]]
// CHECK:     scf.yield %[[ARG1_2]]
// CHECK:   else
// CHECK:     %[[ALLOC:.*]] = memref.alloc
// CHECK:     scf.yield %[[ALLOC]]
// CHECK:   return %[[IF_0]]

func.func @user(%pred : i1, %arg0 : tensor<1x2x3xf32>,
    %arg1 : tensor<1x2x3xf32>, %arg2 : tensor<1x2x3xf32>,
    %arg3 : tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
  %0 = func.call @id(%arg0) : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
  %1 = func.call @ite(%pred, %0, %arg1)
      : (i1, tensor<1x2x3xf32>, tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
  %2 = func.call @ite_select(%pred, %1, %arg2)
      : (i1, tensor<1x2x3xf32>, tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
  %3 = func.call @may_reuse(%pred, %2)
      : (i1, tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
  func.return %3 : tensor<1x2x3xf32>
}

// CHECK: @user(%[[ARG0_4:.*]]: i1, %[[ARG1_3:.*]]: memref<1x2x3xf32>, %[[ARG2_1:.*]]: memref<1x2x3xf32>, %[[ARG3:.*]]: memref<1x2x3xf32>, %[[ARG4:.*]]: memref<1x2x3xf32>) -> memref<1x2x3xf32>
// CHECK:   %[[VAL:.*]] = call @id(%[[ARG1_3]])
// CHECK:   %[[VAL_0:.*]] = call @ite(%[[ARG0_4]], %[[VAL]], %[[ARG2_1]])
// CHECK:   %[[VAL_1:.*]] = call @ite_select(%[[ARG0_4]], %[[VAL_0]], %[[ARG3]])
// CHECK:   %[[VAL_2:.*]] = call @may_reuse(%[[ARG0_4]], %[[VAL_1]])
// CHECK:   return %[[VAL_2]]
