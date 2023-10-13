// RUN: stablehlo-quant-opt %s -split-input-file -stablehlo-prepare-quantize=enable-per-channel-quantization=false -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL: func @dot
func.func @dot(%arg0: tensor<?x3xf32>) -> tensor<?x2xf32> {
  // CHECK: %[[cst:.*]] = stablehlo.constant
  // CHECK: %[[q1:.*]] = "quantfork.qcast"(%[[cst]])
  // CHECK-SAME: quant.uniform<i8:f32, 0.0040316890267764818:127>
  // CHECK: %[[dq1:.*]] = "quantfork.dcast"(%[[q1]])
  // CHECK-SAME: quant.uniform<i8:f32, 0.0040316890267764818:127>
  %cst = stablehlo.constant dense<[[-0.960978984, -0.390246302], [-0.790828585, -0.601039409], [-1.0280807, -1.02731466]]> : tensor<3x2xf32>
  // CHECK: %[[q2:.*]] = "quantfork.qcast"(%arg0)
  // CHECK-SAME: quant.uniform<i8:f32, 0.0078408040252386357:-1>
  // CHECK: %[[dq2:.*]] = "quantfork.dcast"(%[[q2]])
  // CHECK-SAME: quant.uniform<i8:f32, 0.0078408040252386357:-1>
  %0 = "quantfork.stats"(%arg0) {bitsNum = 8 : i64, layerStats = dense<[-0.999415695, 0.99998933]> : tensor<2xf32>, narrowRange = false} : (tensor<?x3xf32>) -> tensor<?x3xf32>
  // CHECK: %[[dot:.*]] = stablehlo.dot %[[dq2]], %[[dq1]]
  %1 = stablehlo.dot %0, %cst : (tensor<?x3xf32>, tensor<3x2xf32>) -> tensor<?x2xf32>
  // CHECK: %[[q3:.*]] = "quantfork.qcast"(%[[dot]])
  // CHECK-SAME: quant.uniform<i8:f32, 0.036254937041039562:-28>>
  // CHECK: %[[dq3:.*]] = "quantfork.dcast"(%[[q3]])
  // CHECK-SAME: quant.uniform<i8:f32, 0.036254937041039562:-28>>
  %2 = "quantfork.stats"(%1) {bitsNum = 8 : i64, layerStats = dense<[-3.6289506, 5.61605835]> : tensor<2xf32>, narrowRange = false} : (tensor<?x2xf32>) -> tensor<?x2xf32>
  // CHECK: return %[[dq3]]
  func.return %2 : tensor<?x2xf32>
}

// -----

// CHECK-LABEL: func @duplicate_stats
func.func @duplicate_stats(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: %[[q1:.*]] = "quantfork.qcast"(%arg0)
  // CHECK: %[[dq1:.*]] = "quantfork.dcast"(%[[q1]])
  // CHECK: %[[q2:.*]] = "quantfork.qcast"(%[[dq1]])
  // CHECK: %[[dq2:.*]] = "quantfork.dcast"(%[[q2]])
  // CHECK: stablehlo.convert %[[dq2]]
  %0 = "quantfork.stats"(%arg0) {bitsNum = 8 : i64, layerStats = dense<[-0.999415695, 0.99998933]> : tensor<2xf32>, narrowRange = false} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "quantfork.stats"(%0) {bitsNum = 8 : i64, layerStats = dense<[-2.0, 2.0]> : tensor<2xf32>, narrowRange = false} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %2 = stablehlo.convert %1 : (tensor<2x3xf32>) -> (tensor<2x3xf32>)
  func.return %2 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: func @dot_redundant_stats
func.func @dot_redundant_stats(%arg0: tensor<?x3xf32>) -> tensor<?x2xf32> {
  // CHECK: %[[cst:.*]] = stablehlo.constant
  // CHECK: %[[q1:.*]] = "quantfork.qcast"(%[[cst]])
  // CHECK-SAME: quant.uniform<i8:f32, 0.0040316890267764818:127>
  // CHECK: %[[dq1:.*]] = "quantfork.dcast"(%[[q1]])
  // CHECK-SAME: quant.uniform<i8:f32, 0.0040316890267764818:127>
  %cst = stablehlo.constant dense<[[-0.960978984, -0.390246302], [-0.790828585, -0.601039409], [-1.0280807, -1.02731466]]> : tensor<3x2xf32>
  // CHECK: %[[q2:.*]] = "quantfork.qcast"(%arg0)
  // CHECK-SAME: quant.uniform<i8:f32, 0.0078408040252386357:-1>
  // CHECK: %[[dq2:.*]] = "quantfork.dcast"(%[[q2]])
  // CHECK-SAME: quant.uniform<i8:f32, 0.0078408040252386357:-1>
  %0 = "quantfork.stats"(%arg0) {bitsNum = 8 : i64, layerStats = dense<[-100.2, 212.4]> : tensor<2xf32>, narrowRange = false} : (tensor<?x3xf32>) -> tensor<?x3xf32>
  %1 = "quantfork.qcast"(%0) {volatile} : (tensor<?x3xf32>) -> tensor<?x3x!quant.uniform<i8:f32, 0.0078408040252386357:-1>>
  %2 = "quantfork.dcast"(%1) : (tensor<?x3x!quant.uniform<i8:f32, 0.0078408040252386357:-1>>) -> tensor<?x3xf32>
  // CHECK: %[[dot:.*]] = stablehlo.dot %[[dq2]], %[[dq1]]
  %3 = stablehlo.dot %2, %cst : (tensor<?x3xf32>, tensor<3x2xf32>) -> tensor<?x2xf32>
  // CHECK: %[[q3:.*]] = "quantfork.qcast"(%[[dot]])
  // CHECK-SAME: quant.uniform<i8:f32, 0.036254937041039562:-28>>
  // CHECK: %[[dq3:.*]] = "quantfork.dcast"(%[[q3]])
  // CHECK-SAME: quant.uniform<i8:f32, 0.036254937041039562:-28>>
  %4 = "quantfork.stats"(%3) {bitsNum = 8 : i64, layerStats = dense<[-3.6289506, 5.61605835]> : tensor<2xf32>, narrowRange = false} : (tensor<?x2xf32>) -> tensor<?x2xf32>
  // CHECK: return %[[dq3]]
  func.return %4 : tensor<?x2xf32>
}

// -----

// CHECK-LABEL: func @convert_same_scale_propagate
func.func @convert_same_scale_propagate(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: %[[dq:.*]] = "quantfork.dcast"
  // CHECK-SAME: (tensor<2x3x!quant.uniform<i8:f32, 0.0078408040252386357:-1>>)
  %0 = "quantfork.stats"(%arg0) {bitsNum = 8 : i64, layerStats = dense<[-0.999415695, 0.99998933]> : tensor<2xf32>, narrowRange = false} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK: %[[convert:.*]] = stablehlo.convert %[[dq]]
  %1 = stablehlo.convert %0 : (tensor<2x3xf32>) -> (tensor<2x3xf32>)
  // CHECK: %[[q:.*]] = "quantfork.qcast"(%[[convert]])
  // CHECK-SAME: -> tensor<2x3x!quant.uniform<i8:f32, 0.0078408040252386357:-1>>
  %2 = "quantfork.stats"(%1) {bitsNum = 8 : i64, layerStats = dense<[-2.0, 2.0]> : tensor<2xf32>, narrowRange = false} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %2 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: func @merge_consecutive_qcast
func.func @merge_consecutive_qcast(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  // CHECK: "quantfork.qcast"(%arg1)
  // CHECK-SAME: -> tensor<*x!quant.uniform<i8:f32, 0.02454993117089365:-64>>
  // CHECK: "quantfork.qcast"(%arg1)
  // CHECK-SAME: -> tensor<*x!quant.uniform<i8:f32, 0.013075299590241675:-64>>
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[-0.83811146, 2.4960899]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
  %1 = "quantfork.stats"(%arg1) {layerStats = dense<[-0.835039615, 1.000000e+00]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
  %2 = "stablehlo.concatenate"(%0, %1) {dimension = 0 : i64} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %3 = "quantfork.stats"(%2) {layerStats = dense<[-0.83811146, 2.4960899]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
  %4 = "quantfork.stats"(%arg2) {layerStats = dense<[-1.5726943, 1.07351148]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
  %5 = "stablehlo.concatenate"(%4, %1) {dimension = 0 : i64} : (tensor<*xf32>,  tensor<*xf32>) -> tensor<*xf32>
  %6 = "quantfork.stats"(%5) {layerStats = dense<[-1.5726943, 4.6875381]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
  func.return  %3, %6 : tensor<*xf32>, tensor<*xf32>
}
