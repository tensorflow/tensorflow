// RUN: tf-tfrt-opt %s -tf-jitrt-legalize-i1-types -split-input-file | FileCheck %s

func.func @func_op(%arg0: tensor<?x?xi1>) -> tensor<?x?xi1> {
  func.return %arg0 : tensor<?x?xi1>
}

// CHECK-LABEL:   func @func_op(
// CHECK-SAME:                  %[[IN_0:.*]]: tensor<?x?xi8>) -> tensor<?x?xi8> {
// CHECK:           return %[[IN_0]] : tensor<?x?xi8>
// CHECK:         }

// -----

func.func @true_constant_op() -> tensor<i1> {
  %0 = mhlo.constant dense<true> : tensor<i1>
  func.return %0 : tensor<i1>
}

// CHECK-LABEL:   func @true_constant_op() -> tensor<i8> {
// CHECK:           %[[TRUE:.*]] = mhlo.constant dense<1> : tensor<i8>
// CHECK:           return %[[TRUE]] : tensor<i8>
// CHECK:         }

// -----

func.func @false_constant_op() -> tensor<i1> {
  %0 = mhlo.constant dense<false> : tensor<i1>
  func.return %0 : tensor<i1>
}

// CHECK-LABEL:   func @false_constant_op() -> tensor<i8> {
// CHECK:           %[[FALSE:.*]] = mhlo.constant dense<0> : tensor<i8>
// CHECK:           return %[[FALSE]] : tensor<i8>
// CHECK:         }

// -----

func.func @and_op(%arg0: tensor<?x?xi1>, %arg1: tensor<?x?xi1>) -> tensor<?x?xi1> {
  %0 = mhlo.and %arg0, %arg1 : tensor<?x?xi1>
  func.return %0 : tensor<?x?xi1>
}

// CHECK-LABEL:   func @and_op(
// CHECK-SAME:                 %[[IN_0:.*]]: tensor<?x?xi8>,
// CHECK-SAME:                 %[[IN_1:.*]]: tensor<?x?xi8>) -> tensor<?x?xi8> {
// CHECK:           %[[AND:.*]] = mhlo.and %[[IN_0]], %[[IN_1]] : tensor<?x?xi8>
// CHECK:           return %[[AND]] : tensor<?x?xi8>
// CHECK:         }

// -----

func.func @or_op(%arg0: tensor<?x?xi1>, %arg1: tensor<?x?xi1>) -> tensor<?x?xi1> {
  %0 = mhlo.or %arg0, %arg1 : tensor<?x?xi1>
  func.return %0 : tensor<?x?xi1>
}

// CHECK-LABEL:   func @or_op(
// CHECK-SAME:                %[[IN_0:.*]]: tensor<?x?xi8>,
// CHECK-SAME:                %[[IN_1:.*]]: tensor<?x?xi8>) -> tensor<?x?xi8> {
// CHECK:           %[[OR:.*]] = mhlo.or %[[IN_0]], %[[IN_1]] : tensor<?x?xi8>
// CHECK:           return %[[OR]] : tensor<?x?xi8>
// CHECK:         }

// -----

func.func @reduce_op(%arg0: tensor<?x?xi1>) -> tensor<?xi1> {
  %0 = mhlo.constant dense<1> : tensor<1xi32>
  %1 = "mhlo.convert"(%arg0) : (tensor<?x?xi1>) -> tensor<?x?xi1>
  %2 = mhlo.constant dense<true> : tensor<i1>
  %3 = "mhlo.reduce"(%1, %2) ({
  ^bb0(%arg1: tensor<i1>, %arg2: tensor<i1>):
    %5 = mhlo.and %arg1, %arg2 : tensor<i1>
    "mhlo.return"(%5) : (tensor<i1>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<?x?xi1>, tensor<i1>) -> tensor<?xi1>
  %4 = "mhlo.convert"(%3) : (tensor<?xi1>) -> tensor<?xi1>
  func.return %4 : tensor<?xi1>
}

// CHECK-LABEL:   func @reduce_op(
// CHECK-SAME:                    %[[IN_0:.*]]: tensor<?x?xi8>) -> tensor<?xi8> {
// CHECK:           %[[TRUE:.*]] = mhlo.constant dense<1> : tensor<i8>

// CHECK:           %[[RED:.*]] = mhlo.reduce(%[[IN_0]] init: %[[TRUE]])
// CHECK:            reducer(%[[ARG_0:.*]]: tensor<i8>, %[[ARG_1:.*]]: tensor<i8>)

// CHECK:           return %[[RED:.*]] : tensor<?xi8>

