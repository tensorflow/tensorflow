// RUN: mlir-hlo-opt -mhlo-test-lower-general-dot -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: @testDebatch1
func.func @testDebatch1(%arg0: tensor<1x1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x1x3xf32> {
  // CHECK-DAG: [[R0:%.+]] = "mhlo.reshape"(%arg0) : (tensor<1x1x2xf32>) -> tensor<1x2xf32>
  // CHECK-DAG: [[R1:%.+]] = "mhlo.dot"([[R0]], %arg1) {precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">]} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
  // CHECK: [[R2:%.+]] = "mhlo.reshape"([[R1]]) : (tensor<1x3xf32>) -> tensor<1x1x3xf32>
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]
    >,
   precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">]
  } : (tensor<1x1x2xf32>, tensor<2x3xf32>) -> tensor<1x1x3xf32>

  func.return %0 : tensor<1x1x3xf32>
}

// -----

// CHECK-LABEL: @testDebatch2
func.func @testDebatch2(%arg0: tensor<2x3xf32>, %arg1: tensor<1x1x2xf32>) -> tensor<3x1x1xf32> {
  // CHECK-DAG: [[R0:%.+]] = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  // CHECK-DAG: [[R1:%.+]] = "mhlo.transpose"(%arg1) {permutation = dense<[2, 0, 1]> : tensor<3xi64>} : (tensor<1x1x2xf32>) -> tensor<2x1x1xf32>
  // CHECK-DAG: [[R2:%.+]] = "mhlo.reshape"([[R1]]) : (tensor<2x1x1xf32>) -> tensor<2x1xf32>
  // CHECK-DAG: [[R3:%.+]] = "mhlo.dot"([[R0]], [[R2]]) {precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">]} : (tensor<3x2xf32>, tensor<2x1xf32>) -> tensor<3x1xf32>
  // CHECK: [[R4:%.+]] = "mhlo.reshape"([[R3]]) : (tensor<3x1xf32>) -> tensor<3x1x1xf32>

  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_contracting_dimensions = [0],
      rhs_contracting_dimensions = [2]
    >,
    precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">]
  } : (tensor<2x3xf32>, tensor<1x1x2xf32>) -> tensor<3x1x1xf32>
  func.return %0 : tensor<3x1x1xf32>
}

// -----

// CHECK-LABEL: @testBatchPassthrough
func.func @testBatchPassthrough(%arg0: tensor<2x2x3xf32>, %arg1: tensor<2x1x2xf32>) -> tensor<3x2x1xf32> {
  // CHECK-NEXT: "mhlo.dot_general"(%arg0, %arg1)
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [2]
    >,
    precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">]
  } : (tensor<2x2x3xf32>, tensor<2x1x2xf32>) -> tensor<3x2x1xf32>
  func.return %0 : tensor<3x2x1xf32>
}

// -----

// CHECK-LABEL: @testVec
func.func @testVec(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) -> tensor<f32> {
  // CHECK-NEXT: [[R0:%.+]] = "mhlo.reshape"(%arg0) : (tensor<32xf32>) -> tensor<1x32xf32>
  // CHECK-NEXT: [[R1:%.+]] = "mhlo.reshape"(%arg1) : (tensor<32xf32>) -> tensor<32x1xf32>
  // CHECK-NEXT: [[M:%.+]] = "mhlo.dot"([[R0]], [[R1]])
  // CHECK-NEXT: [[RR:%.+]] = "mhlo.reshape"([[M]]) : (tensor<1x1xf32>) -> tensor<f32>
  // CHECK-NEXT: return [[RR]]
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_contracting_dimensions = [0],
      rhs_contracting_dimensions = [0]
    >,
    precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">]
  } : (tensor<32xf32>, tensor<32xf32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @testMatVec
func.func @testMatVec(%arg0: tensor<32x20xf32>, %arg1: tensor<32xf32>) -> tensor<20xf32> {
  // CHECK-NEXT: [[T:%.+]] = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]>
  // CHECK-NEXT: [[R1:%.+]] = "mhlo.reshape"(%arg1) : (tensor<32xf32>) -> tensor<32x1xf32>
  // CHECK-NEXT: [[M:%.+]] = "mhlo.dot"([[T]], [[R1]])
  // CHECK-NEXT: [[R:%.+]] = "mhlo.reshape"([[M]])
  // CHECK-NEXT: return [[R]]
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_contracting_dimensions = [0],
      rhs_contracting_dimensions = [0]
    >,
    precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">]
  } : (tensor<32x20xf32>, tensor<32xf32>) -> tensor<20xf32>
  func.return %0 : tensor<20xf32>
}

// -----

func.func @dot_general_to_dot_dynamic(%arg0: tensor<128x4x?x32xf32>, %arg1: tensor<8x?x128x4xf32>) -> tensor<?x32x8x?xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [],
      lhs_contracting_dimensions = [0, 1],
      rhs_batching_dimensions = [],
      rhs_contracting_dimensions = [2, 3],
    >,
    precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">]
  } : (tensor<128x4x?x32xf32>, tensor<8x?x128x4xf32>) -> tensor<?x32x8x?xf32>
  func.return %0 : tensor<?x32x8x?xf32>
}
// CHECK-LABEL: func @dot_general_to_dot_dynamic
// CHECK-DAG: %[[C32:.+]] = mhlo.constant dense<32> : tensor<1xi32>
// CHECK-DAG: %[[C512:.+]] = mhlo.constant dense<512> : tensor<1xi32>
// CHECK-DAG: %[[C8:.+]] = mhlo.constant dense<8> : tensor<1xi32>
// CHECK-DAG: %[[TRANS0:.+]] = "mhlo.transpose"(%arg0) {permutation = dense<[2, 3, 0, 1]> : tensor<4xi64>}
// CHECK-DAG: %[[DIM0:.+]] = "mhlo.get_dimension_size"(%arg0) {dimension = 2 : i64}
// CHECK-DAG: %[[MUL0:.+]] = mhlo.multiply %[[DIM0]], %[[C32]]
// CHECK-DAG: %[[CONCAT1:.+]] = "mhlo.concatenate"(%[[MUL0]], %[[C512]]) {dimension = 0 : i64}
// CHECK-DAG: %[[DR1:.+]] = "mhlo.dynamic_reshape"(%[[TRANS0]], %[[CONCAT1]])
// CHECK-DAG: %[[TRANS1:.+]] = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 0, 1]> : tensor<4xi64>}
// CHECK-DAG: %[[DIM1:.+]] = "mhlo.get_dimension_size"(%arg1) {dimension = 1 : i64}
// CHECK-DAG: %[[MUL1:.+]] = mhlo.multiply %[[DIM1]], %[[C8]]
// CHECK-DAG: %[[CONCAT2:.+]] = "mhlo.concatenate"(%[[C512]], %[[MUL1]]) {dimension = 0 : i64}
// CHECK-DAG: %[[DR2:.+]] = "mhlo.dynamic_reshape"(%[[TRANS1]], %[[CONCAT2]])
// CHECK-DAG: %[[DOT:.+]] = "mhlo.dot"(%[[DR1:.+]], %[[DR2:.+]])
// CHECK-DAG: %[[DIM2:.+]] = "mhlo.get_dimension_size"(%arg0) {dimension = 2 : i64}
// CHECK-DAG: %[[DIM3:.+]] = "mhlo.get_dimension_size"(%arg1) {dimension = 1 : i64}
// CHECK-DAG: %[[CONCAT3:.+]] = "mhlo.concatenate"(%[[DIM2]], %[[C32]], %[[C8]], %[[DIM3]]) {dimension = 0 : i64}
// CHECK-DAG: %[[DR3:.+]] = "mhlo.dynamic_reshape"(%[[DOT]], %[[CONCAT3]])
// CHECK: return %[[DR3]]


// -----

func.func @dot_no_rhs_batch(%arg0: tensor<1x512x768xf32>, %arg1: tensor<768x12x64xf32>) -> tensor<1x512x12x64xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [0]>
    } : (tensor<1x512x768xf32>, tensor<768x12x64xf32>) -> tensor<1x512x12x64xf32>
  func.return %0 : tensor<1x512x12x64xf32>
}

// CHECK-LABEL:  func @dot_no_rhs_batch
// CHECK:          %[[RESHAPEL:.+]] = "mhlo.reshape"(%arg0) : (tensor<1x512x768xf32>) -> tensor<512x768xf32>
// CHECK:          %[[RESHAPER:.+]] = "mhlo.reshape"(%arg1) : (tensor<768x12x64xf32>) -> tensor<768x768xf32>
// CHECK:          %[[DOT:.+]] = "mhlo.dot"(%[[RESHAPEL]], %[[RESHAPER]]) : (tensor<512x768xf32>, tensor<768x768xf32>) -> tensor<512x768xf32>
// CHECK:          %[[OUT:.+]] = "mhlo.reshape"(%[[DOT]]) : (tensor<512x768xf32>) -> tensor<1x512x12x64xf32>
