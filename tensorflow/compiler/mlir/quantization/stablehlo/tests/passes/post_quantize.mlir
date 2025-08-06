// RUN: stablehlo-quant-opt %s -split-input-file -tf-stablehlo-post-quantize | FileCheck %s

// CHECK-LABEL: @remove_volatile_qdq
func.func @remove_volatile_qdq() -> tensor<3x2xf32> {
  // CHECK: %[[CST:.*]] = stablehlo.constant
  // CHECK-NOT: "quantization.qcast"
  // CHECK-NOT: "quantization.dcast"
  // CHECK: return %[[CST]]
  %cst = stablehlo.constant dense<[[-0.960978984, -0.390246302], [-0.790828585, -0.601039409], [-1.0280807, -1.02731466]]> : tensor<3x2xf32>
  %q = "quantization.qcast"(%cst) {volatile} : (tensor<3x2xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 0.013075299590241675:-64>>
  %dq = "quantization.dcast"(%q) : (tensor<3x2x!quant.uniform<i8:f32, 0.013075299590241675:-64>>) -> tensor<3x2xf32>
  func.return %dq : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @remove_volatile_qdq_with_requantization
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x2xf32>
func.func @remove_volatile_qdq_with_requantization(%arg0: tensor<3x2xf32>) -> tensor<3x2xf32> {
  // CHECK: %[[Q1:.*]] = stablehlo.uniform_quantize %[[ARG0]]
  // CHECK: %[[Q2:.*]] = stablehlo.uniform_quantize %[[Q1]]
  // CHECK: %[[ABS:.*]] = stablehlo.abs %[[Q2]]
  // CHECK: %[[DQ:.*]] = stablehlo.uniform_dequantize %[[ABS]]
  // CHECK: %[[ADD:.*]] = stablehlo.add %[[ARG0]], %[[DQ]]
  // CHECK: return %[[ADD]]
  %q1 = "quantization.qcast"(%arg0) {volatile} : (tensor<3x2xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 6.000000e-03:-128>>
  %q2 = "quantization.qcast"(%q1) {volatile} : (tensor<3x2x!quant.uniform<i8:f32, 6.000000e-03:-128>>) -> tensor<3x2x!quant.uniform<i8:f32, 0.013075299590241675:-64>>
  %dq1 = "quantization.dcast"(%q2) : (tensor<3x2x!quant.uniform<i8:f32, 0.013075299590241675:-64>>) -> tensor<3x2xf32>
  %abs = stablehlo.abs %q2 : (tensor<3x2x!quant.uniform<i8:f32, 0.013075299590241675:-64>>) -> tensor<3x2x!quant.uniform<i8:f32, 0.013075299590241675:-64>>
  %dq2 = "quantization.dcast"(%abs) : (tensor<3x2x!quant.uniform<i8:f32, 0.013075299590241675:-64>>) -> tensor<3x2xf32>
  %add = stablehlo.add %dq1, %dq2 : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
  func.return %add : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @quantize_constant
// CHECK-SAME: %[[ARG0:.*]]: tensor<1x3xf32>
func.func @quantize_constant(%arg0: tensor<1x3xf32>) -> tensor<1x2xf32> {
  // CHECK-DAG: %[[QCST:.*]] = stablehlo.constant() <{value = dense<-78> : tensor<3x2xi8>}> : () -> tensor<3x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  // CHECK-DAG: %[[Q1:.*]] = stablehlo.uniform_quantize %[[ARG0]]
  // CHECK-NOT: "quantization.qcast"
  // CHECK: %[[DOT:.*]] = stablehlo.dot %[[Q1]], %[[QCST]]
  // CHECK: %[[DQ:.*]] = stablehlo.uniform_dequantize %[[DOT]]
  // CHECK: return %[[DQ]]
  %cst = stablehlo.constant dense<-0.390246302> : tensor<3x2xf32>
  %q1 = "quantization.qcast"(%arg0) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 6.000000e-03:-128>>
  %q2 = "quantization.qcast"(%cst) {volatile} : (tensor<3x2xf32>) -> tensor<3x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  %dot = stablehlo.dot %q1, %q2 : (tensor<1x3x!quant.uniform<i8:f32, 6.000000e-03:-128>>, tensor<3x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>) -> tensor<1x2x!quant.uniform<i8:f32, 1.000000e-03:-3>>
  %dq = "quantization.dcast"(%dot) : (tensor<1x2x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<1x2xf32>
  func.return %dq : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: @convert_quantization_qdq_to_stablehlo_uniform_qdq
// CHECK-SAME: %[[ARG0:.*]]: tensor<1x3xf32>
// CHECK-SAME: %[[ARG1:.*]]: tensor<3x2xf32>
func.func @convert_quantization_qdq_to_stablehlo_uniform_qdq(%arg0: tensor<1x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<1x2xf32> {
  // CHECK: %[[Q1:.*]] = stablehlo.uniform_quantize %[[ARG0]]
  // CHECK-NOT: "quantization.qcast"
  // CHECK: %[[Q2:.*]] = stablehlo.uniform_quantize %[[ARG1]]
  // CHECK-NOT: "quantization.qcast"
  // CHECK: %[[DOT:.*]] = stablehlo.dot %[[Q1]], %[[Q2]]
  // CHECK: %[[DQ:.*]] = stablehlo.uniform_dequantize %[[DOT]]
  // CHECK: return %[[DQ]]
  %q1 = "quantization.qcast"(%arg0) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 6.000000e-03:-128>>
  %q2 = "quantization.qcast"(%arg1) {volatile} : (tensor<3x2xf32>) -> tensor<3x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  %dot = stablehlo.dot %q1, %q2 : (tensor<1x3x!quant.uniform<i8:f32, 6.000000e-03:-128>>, tensor<3x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>) -> tensor<1x2x!quant.uniform<i8:f32, 1.000000e-03:-3>>
  %dq = "quantization.dcast"(%dot) : (tensor<1x2x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<1x2xf32>
  func.return %dq : tensor<1x2xf32>
}
