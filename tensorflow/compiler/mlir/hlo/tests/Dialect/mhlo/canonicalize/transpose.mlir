// RUN: mlir-hlo-opt %s -split-input-file -pass-pipeline='func.func(canonicalize)' | FileCheck %s

// CHECK-LABEL: func @transpose_splat_constant
func.func @transpose_splat_constant() -> tensor<5x10xf32> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<1.000000e+00> : tensor<5x10xf32>
  %cst = mhlo.constant dense<1.000000e+00> : tensor<10x5xf32>
  %0 = "mhlo.transpose"(%cst) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<10x5xf32>) -> tensor<5x10xf32>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<5x10xf32>
}

// -----

// CHECK-LABEL: func @remove_noop
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @remove_noop(%arg : tensor<2x3x9x5xi32>) -> tensor<2x3x9x5xi32> {
  %0 = "mhlo.transpose"(%arg) {permutation = dense<[0, 1, 2, 3]> : tensor<4xi64>}: (tensor<2x3x9x5xi32>) -> tensor<2x3x9x5xi32>
  // CHECK-NEXT: return [[ARG]]
  func.return %0 : tensor<2x3x9x5xi32>
}

// -----

// CHECK-LABEL: func @keep_real_transpose
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @keep_real_transpose(%arg : tensor<2x3x9x5xi32>) -> tensor<3x2x5x9xi32> {
  // CHECK-NEXT: "mhlo.transpose"([[ARG]])
  %0 = "mhlo.transpose"(%arg) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>}: (tensor<2x3x9x5xi32>) -> tensor<3x2x5x9xi32>
  func.return %0 : tensor<3x2x5x9xi32>
}

// -----

// CHECK-LABEL: func @keep_same_shape_real_transpose
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @keep_same_shape_real_transpose(%arg : tensor<4x4xi32>) -> tensor<4x4xi32> {
  // CHECK-NEXT: "mhlo.transpose"([[ARG]])
  %0 = "mhlo.transpose"(%arg) {permutation = dense<[1, 0]> : tensor<2xi64>}: (tensor<4x4xi32>) -> tensor<4x4xi32>
  func.return %0 : tensor<4x4xi32>
}

// -----

// CHECK-LABEL: @eliminate_redundant_transpose
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @eliminate_redundant_transpose(%arg : tensor<3x4x16x2xf32>) -> tensor<3x2x16x4xf32> {
  %0 = "mhlo.transpose"(%arg) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>}: (tensor<3x4x16x2xf32>) -> tensor<3x2x4x16xf32>
  %1 = "mhlo.transpose"(%0) {permutation = dense<[0, 1, 3, 2]> : tensor<4xi64>}: (tensor<3x2x4x16xf32>) -> tensor<3x2x16x4xf32>
  // CHECK: [[RET:%[a-zA-Z0-9]+]] = "mhlo.transpose"([[ARG]])
  // CHECK-SAME: dense<[0, 3, 2, 1]
  // CHECK-NEXT: return [[RET]]
  func.return %1 : tensor<3x2x16x4xf32>
}

// -----

// CHECK-LABEL: func @broadcast_transpose
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @broadcast_transpose(%arg0 : tensor<64xf32>) -> tensor<5x64x31x95xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<5x31x95x64xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<5x31x95x64xf32>) -> tensor<5x64x31x95xf32>
    // CHECK: [[RET:%[a-zA-Z0-9]+]] = "mhlo.broadcast_in_dim"([[ARG]])
    // CHECK-SAME: dense<1>
    // CHECK-NEXT: return [[RET]]
    func.return %1 : tensor<5x64x31x95xf32>
}

// -----

// CHECK-LABEL: func @broadcast_transpose_non_dim
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @broadcast_transpose_non_dim(%arg0 : tensor<f32>) -> tensor<5x64x31x95xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<5x31x95x64xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<5x31x95x64xf32>) -> tensor<5x64x31x95xf32>
    // CHECK: [[RET:%[a-zA-Z0-9]+]] = "mhlo.broadcast_in_dim"([[ARG]])
    // CHECK-SAME: dense<>
    // CHECK-NEXT: return [[RET]]
    func.return %1 : tensor<5x64x31x95xf32>
}

// -----

// CHECK-LABEL: func @broadcast_transpose_multi_dim
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @broadcast_transpose_multi_dim(%arg0 : tensor<95x64xf32>) -> tensor<5x64x31x95xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[2, 3]> : tensor<2xi64>} : (tensor<95x64xf32>) -> tensor<5x31x95x64xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<5x31x95x64xf32>) -> tensor<5x64x31x95xf32>
    // CHECK: [[RET:%[a-zA-Z0-9]+]] = "mhlo.broadcast_in_dim"([[ARG]])
    // CHECK-SAME: dense<[3, 1]>
    // CHECK-NEXT: return [[RET]]
    func.return %1 : tensor<5x64x31x95xf32>
}
