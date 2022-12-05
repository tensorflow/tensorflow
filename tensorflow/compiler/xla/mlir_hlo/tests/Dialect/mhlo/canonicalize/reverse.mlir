// RUN: mlir-hlo-opt %s -pass-pipeline='builtin.module(func.func(canonicalize))' | FileCheck %s

// CHECK-LABEL: func @noop
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x2xf32>)
func.func @noop(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %0 = "mhlo.reverse"(%arg0) {dimensions = dense<[]> : tensor<0xi64>} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  // CHECK: return %[[ARG0]]
  func.return %0 : tensor<1x2xf32>
}

// CHECK-LABEL: func @dim1
// CHECK-SAME: (%[[ARG0:.*]]: tensor
func.func @dim1(%arg0: tensor<9x1x2x1x42xf32>) -> tensor<9x1x2x1x42xf32> {
  %0 = "mhlo.reverse"(%arg0) {dimensions = dense<[1,3]> : tensor<2xi64>} : (tensor<9x1x2x1x42xf32>) -> tensor<9x1x2x1x42xf32>
  // CHECK: return %[[ARG0]]
  func.return %0 : tensor<9x1x2x1x42xf32>
}

// CHECK-LABEL: @noop_reverse_dynamic_shape
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @noop_reverse_dynamic_shape(%arg0 : tensor<10x?x512xf32>) -> tensor<10x?x512xf32> {
  %0 = "mhlo.reverse"(%arg0) {dimensions = dense<[0,1]> : tensor<2xi64>}: (tensor<10x?x512xf32>) -> tensor<10x?x512xf32>
  // CHECK-NEXT: "mhlo.reverse"([[ARG]])
  func.return %0 : tensor<10x?x512xf32>
}

// CHECK-LABEL: func @reverse_fold_constant_int
func.func @reverse_fold_constant_int() -> tensor<0x2x0xi64> {
  %cst = mhlo.constant dense<> : tensor<0x2x0xi64>
  // CHECK: mhlo.constant dense<>
  %1 = "mhlo.reverse"(%cst) {dimensions = dense<[0,1]> : tensor<2xi64>} : (tensor<0x2x0xi64>) -> tensor<0x2x0xi64>
  func.return %1 : tensor<0x2x0xi64>
}

// CHECK-LABEL: func @reverse_fold_constant_int_0
func.func @reverse_fold_constant_int_0() -> tensor<0xi64> {
  %cst = mhlo.constant dense<> : tensor<0xi64>
  // CHECK: mhlo.constant dense<>
  %1 = "mhlo.reverse"(%cst) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<0xi64>) -> tensor<0xi64>
  func.return %1 : tensor<0xi64>
}

// CHECK-LABEL: func @reverse_fold_constant_int_1
func.func @reverse_fold_constant_int_1() -> tensor<3x2xi32> {
  %cst = mhlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
  // CHECK: mhlo.constant dense<{{\[\[}}6, 5], [4, 3], [2, 1]]>
  %1 = "mhlo.reverse"(%cst) {dimensions = dense<[0,1]> : tensor<2xi64>} : (tensor<3x2xi32>) -> tensor<3x2xi32>
  func.return %1 : tensor<3x2xi32>
}

// CHECK-LABEL: func @reverse_fold_constant_int_2
func.func @reverse_fold_constant_int_2() -> tensor<3x2xi32> {
  %cst = mhlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
  // CHECK: mhlo.constant dense<{{\[\[}}5, 6], [3, 4], [1, 2]]>
  %1 = "mhlo.reverse"(%cst) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<3x2xi32>) -> tensor<3x2xi32>
  func.return %1 : tensor<3x2xi32>
}

// CHECK-LABEL: func @reverse_fold_constant_int_3
func.func @reverse_fold_constant_int_3() -> tensor<3x2xi32> {
  %cst = mhlo.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
  // CHECK: mhlo.constant dense<{{\[\[}}2, 1], [4, 3], [6, 5]]>
  %1 = "mhlo.reverse"(%cst) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<3x2xi32>) -> tensor<3x2xi32>
  func.return %1 : tensor<3x2xi32>
}

// CHECK-LABEL: func @reverse_fold_constant_int_4
func.func @reverse_fold_constant_int_4() -> tensor<2x3x2xi32> {
  %cst = mhlo.constant dense<[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]> : tensor<2x3x2xi32>
  // CHECK: mhlo.constant dense<{{\[\[\[}}12, 11], [10, 9], [8, 7]], {{\[\[}}6, 5], [4, 3], [2, 1]]]>
  %1 = "mhlo.reverse"(%cst) {dimensions = dense<[0,1,2]> : tensor<3xi64>} : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
  func.return %1 : tensor<2x3x2xi32>
}

// CHECK-LABEL: func @reverse_fold_constant_float
func.func @reverse_fold_constant_float() -> tensor<3x2xf32> {
  %cst = mhlo.constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32>
  // CHECK: mhlo.constant dense<{{\[\[}}6.000000e+00, 5.000000e+00], [4.000000e+00, 3.000000e+00], [2.000000e+00, 1.000000e+00]]>
  %1 = "mhlo.reverse"(%cst) {dimensions = dense<[0,1]> : tensor<2xi64>} : (tensor<3x2xf32>) -> tensor<3x2xf32>
  func.return %1 : tensor<3x2xf32>
}

