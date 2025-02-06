// RUN: mlir-hlo-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(canonicalize))' | FileCheck %s

// CHECK-LABEL: func @transpose_splat_constant
func.func @transpose_splat_constant() -> tensor<5x10xf32> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<1.000000e+00> : tensor<5x10xf32>
  %cst = mhlo.constant dense<1.000000e+00> : tensor<10x5xf32>
  %0 = "mhlo.transpose"(%cst) <{permutation = dense<[1, 0]> : tensor<2xi64>}> : (tensor<10x5xf32>) -> tensor<5x10xf32>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<5x10xf32>
}

// -----

// CHECK-LABEL: func @remove_noop
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @remove_noop(%arg : tensor<2x3x9x5xi32>) -> tensor<2x3x9x5xi32> {
  %0 = "mhlo.transpose"(%arg) <{permutation = dense<[0, 1, 2, 3]> : tensor<4xi64>}>: (tensor<2x3x9x5xi32>) -> tensor<2x3x9x5xi32>
  // CHECK-NEXT: return [[ARG]]
  func.return %0 : tensor<2x3x9x5xi32>
}

// -----

// CHECK-LABEL: func @keep_real_transpose
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @keep_real_transpose(%arg : tensor<2x3x9x5xi32>) -> tensor<3x2x5x9xi32> {
  // CHECK-NEXT: "mhlo.transpose"([[ARG]])
  %0 = "mhlo.transpose"(%arg) <{permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>}>: (tensor<2x3x9x5xi32>) -> tensor<3x2x5x9xi32>
  func.return %0 : tensor<3x2x5x9xi32>
}

// -----

// CHECK-LABEL: func @keep_same_shape_real_transpose
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @keep_same_shape_real_transpose(%arg : tensor<4x4xi32>) -> tensor<4x4xi32> {
  // CHECK-NEXT: "mhlo.transpose"([[ARG]])
  %0 = "mhlo.transpose"(%arg) <{permutation = dense<[1, 0]> : tensor<2xi64>}>: (tensor<4x4xi32>) -> tensor<4x4xi32>
  func.return %0 : tensor<4x4xi32>
}

// -----

// CHECK-LABEL: @eliminate_redundant_transpose
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @eliminate_redundant_transpose(%arg : tensor<3x4x16x2xf32>) -> tensor<3x2x16x4xf32> {
  %0 = "mhlo.transpose"(%arg) <{permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>}>: (tensor<3x4x16x2xf32>) -> tensor<3x2x4x16xf32>
  %1 = "mhlo.transpose"(%0) <{permutation = dense<[0, 1, 3, 2]> : tensor<4xi64>}>: (tensor<3x2x4x16xf32>) -> tensor<3x2x16x4xf32>
  // CHECK: [[RET:%[a-zA-Z0-9]+]] = "mhlo.transpose"([[ARG]])
  // CHECK-SAME: dense<[0, 3, 2, 1]
  // CHECK-NEXT: return [[RET]]
  func.return %1 : tensor<3x2x16x4xf32>
}

// -----

// CHECK-LABEL: @simplify_transpose_case1
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @simplify_transpose_case1(%arg : tensor<10x1x512xf32>) -> tensor<1x10x512xf32> {
  %0 = "mhlo.transpose"(%arg) <{permutation = dense<[1, 0, 2]> : tensor<3xi64>}>: (tensor<10x1x512xf32>) -> tensor<1x10x512xf32>
  // CHECK-NEXT: mhlo.reshape [[ARG]]
  func.return %0 : tensor<1x10x512xf32>
}

// -----

// CHECK-LABEL: @simplify_transpose_case2
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @simplify_transpose_case2(%arg : tensor<10x1x512x1xf32>) -> tensor<1x1x10x512xf32> {
  %0 = "mhlo.transpose"(%arg) <{permutation = dense<[1, 3, 0, 2]> : tensor<4xi64>}>: (tensor<10x1x512x1xf32>) -> tensor<1x1x10x512xf32>
  // CHECK-NEXT: mhlo.reshape [[ARG]]
  func.return %0 : tensor<1x1x10x512xf32>
}

// -----

// CHECK-LABEL: @not_simplify_transpose_dynamic_shape
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @not_simplify_transpose_dynamic_shape(%arg : tensor<10x?x512xf32>) -> tensor<?x10x512xf32> {
  %0 = "mhlo.transpose"(%arg) <{permutation = dense<[1, 0, 2]> : tensor<3xi64>}>: (tensor<10x?x512xf32>) -> tensor<?x10x512xf32>
  // CHECK-NEXT: "mhlo.transpose"([[ARG]])
  func.return %0 : tensor<?x10x512xf32>
}

// -----

// CHECK-LABEL: func @broadcast_transpose
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @broadcast_transpose(%arg0 : tensor<64xf32>) -> tensor<5x64x31x95xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<3> : tensor<1xi64>}> : (tensor<64xf32>) -> tensor<5x31x95x64xf32>
    %1 = "mhlo.transpose"(%0) <{permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>}> : (tensor<5x31x95x64xf32>) -> tensor<5x64x31x95xf32>
    // CHECK: [[RET:%[a-zA-Z0-9]+]] = "mhlo.broadcast_in_dim"([[ARG]])
    // CHECK-SAME: dense<1>
    // CHECK-NEXT: return [[RET]]
    func.return %1 : tensor<5x64x31x95xf32>
}

// -----

// CHECK-LABEL: func @broadcast_transpose_non_dim
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @broadcast_transpose_non_dim(%arg0 : tensor<f32>) -> tensor<5x64x31x95xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<5x31x95x64xf32>
    %1 = "mhlo.transpose"(%0) <{permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>}> : (tensor<5x31x95x64xf32>) -> tensor<5x64x31x95xf32>
    // CHECK: [[RET:%[a-zA-Z0-9]+]] = "mhlo.broadcast_in_dim"([[ARG]])
    // CHECK-SAME: dense<>
    // CHECK-NEXT: return [[RET]]
    func.return %1 : tensor<5x64x31x95xf32>
}

// -----

// CHECK-LABEL: func @broadcast_transpose_multi_dim
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @broadcast_transpose_multi_dim(%arg0 : tensor<95x64xf32>) -> tensor<5x64x31x95xf32> {
    %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[2, 3]> : tensor<2xi64>}> : (tensor<95x64xf32>) -> tensor<5x31x95x64xf32>
    %1 = "mhlo.transpose"(%0) <{permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>}> : (tensor<5x31x95x64xf32>) -> tensor<5x64x31x95xf32>
    // CHECK: [[RET:%[a-zA-Z0-9]+]] = "mhlo.broadcast_in_dim"([[ARG]])
    // CHECK-SAME: dense<[3, 1]>
    // CHECK-NEXT: return [[RET]]
    func.return %1 : tensor<5x64x31x95xf32>
}

// -----

// CHECK-LABEL: func @transpose_splat_constant_quantized_per_tensor
// CHECK-NOT: mhlo.transpose
func.func @transpose_splat_constant_quantized_per_tensor() -> tensor<5x10x!quant.uniform<i8:f32, 2.000000e+0:16>> {
  %cst = mhlo.constant() {value = dense<42> : tensor<10x5xi8>} : () -> tensor<10x5x!quant.uniform<i8:f32, 2.000000e+0:16>>
  %0 = "mhlo.transpose"(%cst) <{permutation = dense<[1, 0]> : tensor<2xi64>}> : (tensor<10x5x!quant.uniform<i8:f32, 2.000000e+0:16>>) -> tensor<5x10x!quant.uniform<i8:f32, 2.000000e+0:16>>
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant
  // CHECK-SAME: tensor<5x10x!quant.uniform<i8:f32, 2.000000e+00:16>>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<5x10x!quant.uniform<i8:f32, 2.000000e+0:16>>
}

// -----

// CHECK-LABEL: func @transpose_splat_constant_quantized_per_axis
// CHECK-NOT: mhlo.transpose
func.func @transpose_splat_constant_quantized_per_axis() -> tensor<2x10x!quant.uniform<i8:f32:0, {2.000000e+0:16,3.000000e+0:32}>> {
  %cst = mhlo.constant() {value = dense<42> : tensor<10x2xi8>} : () -> tensor<10x2x!quant.uniform<i8:f32:1, {2.000000e+0:16,3.000000e+0:32}>>
  %0 = "mhlo.transpose"(%cst) <{permutation = dense<[1, 0]> : tensor<2xi64>}> : (tensor<10x2x!quant.uniform<i8:f32:1, {2.000000e+0:16,3.000000e+0:32}>>) -> tensor<2x10x!quant.uniform<i8:f32:0, {2.000000e+0:16,3.000000e+0:32}>>
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant
  // CHECK-SAME: tensor<2x10x!quant.uniform<i8:f32:0, {2.000000e+00:16,3.000000e+00:32}>>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<2x10x!quant.uniform<i8:f32:0, {2.000000e+0:16,3.000000e+0:32}>>
}

// -----

// Can not fold non-splat tensors (quantized or not)
// CHECK-LABEL: func @nofold_nonsplat_quant_constant
func.func @nofold_nonsplat_quant_constant() -> tensor<4x2x!quant.uniform<i8:f32, 2.000000e+0:16>> {
  %cst = mhlo.constant() {value = dense<[[1, 2, 3, 4],[5, 6, 7, 8]]> : tensor<2x4xi8>} : () -> tensor<2x4x!quant.uniform<i8:f32, 2.000000e+0:16>>
  %0 = "mhlo.transpose"(%cst) <{permutation = dense<[1, 0]> : tensor<2xi64>}> : (tensor<2x4x!quant.uniform<i8:f32, 2.000000e+0:16>>) -> tensor<4x2x!quant.uniform<i8:f32, 2.000000e+0:16>>
  // CHECK: [[TRANSPOSED:%.+]] = "mhlo.transpose"
  // CHECK-SAME: -> tensor<4x2x!quant.uniform<i8:f32, 2.000000e+00:16>>
  // CHECK-NEXT: return [[TRANSPOSED]]
  func.return %0 : tensor<4x2x!quant.uniform<i8:f32, 2.000000e+0:16>>
}
