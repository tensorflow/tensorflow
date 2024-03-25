// RUN: stablehlo-quant-opt %s -split-input-file -stablehlo-prepare-quantize=enable-per-channel-quantized-weight=false -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL: func @dot
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<?x3xf32>) -> tensor<?x2xf32>
func.func @dot(%arg0: tensor<?x3xf32>) -> tensor<?x2xf32> {
  // CHECK: %[[cst:.*]] = stablehlo.constant
  // CHECK: %[[q1:.*]] = "quantfork.qcast"(%[[cst]])
  // CHECK-SAME: quant.uniform<i8:f32, 0.0040316890267764818:127>
  // CHECK: %[[dq1:.*]] = "quantfork.dcast"(%[[q1]])
  // CHECK-SAME: quant.uniform<i8:f32, 0.0040316890267764818:127>
  %cst = stablehlo.constant dense<[[-0.960978984, -0.390246302], [-0.790828585, -0.601039409], [-1.0280807, -1.02731466]]> : tensor<3x2xf32>
  // CHECK: %[[q2:.*]] = "quantfork.qcast"(%[[ARG_0]])
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
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<2x3xf32>) -> tensor<2x3xf32>
func.func @duplicate_stats(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: %[[q1:.*]] = "quantfork.qcast"(%[[ARG_0]])
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
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<?x3xf32>) -> tensor<?x2xf32>
func.func @dot_redundant_stats(%arg0: tensor<?x3xf32>) -> tensor<?x2xf32> {
  // CHECK: %[[cst:.*]] = stablehlo.constant
  // CHECK: %[[q1:.*]] = "quantfork.qcast"(%[[cst]])
  // CHECK-SAME: quant.uniform<i8:f32, 0.0040316890267764818:127>
  // CHECK: %[[dq1:.*]] = "quantfork.dcast"(%[[q1]])
  // CHECK-SAME: quant.uniform<i8:f32, 0.0040316890267764818:127>
  %cst = stablehlo.constant dense<[[-0.960978984, -0.390246302], [-0.790828585, -0.601039409], [-1.0280807, -1.02731466]]> : tensor<3x2xf32>
  // CHECK: %[[q2:.*]] = "quantfork.qcast"(%[[ARG_0]])
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

// CHECK-LABEL: func @reshape_same_scale_propagate
func.func @reshape_same_scale_propagate(%arg0: tensor<2x3xf32>) -> tensor<6xf32> {
  // CHECK: %[[dq:.*]] = "quantfork.dcast"
  // CHECK-SAME: (tensor<2x3x!quant.uniform<i8:f32, 0.0078408040252386357:-1>>)
  %0 = "quantfork.stats"(%arg0) {bitsNum = 8 : i64, layerStats = dense<[-0.999415695, 0.99998933]> : tensor<2xf32>, narrowRange = false} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK: %[[reshape:.*]] = stablehlo.reshape %[[dq]]
  %1 = stablehlo.reshape %0 : (tensor<2x3xf32>) -> (tensor<6xf32>)
  // CHECK: %[[q:.*]] = "quantfork.qcast"(%[[reshape]])
  // CHECK-SAME: -> tensor<6x!quant.uniform<i8:f32, 0.0078408040252386357:-1>>
  %2 = "quantfork.stats"(%1) {bitsNum = 8 : i64, layerStats = dense<[-2.0, 2.0]> : tensor<2xf32>, narrowRange = false} : (tensor<6xf32>) -> tensor<6xf32>
  func.return %2 : tensor<6xf32>
}

// -----

// CHECK-LABEL: func @merge_consecutive_qcast
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<?xf32>, %[[ARG_1:.*]]: tensor<?xf32>, %[[ARG_2:.*]]: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>)
func.func @merge_consecutive_qcast(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  // CHECK: "quantfork.qcast"(%[[ARG_1]])
  // CHECK-SAME: -> tensor<?x!quant.uniform<i8:f32, 0.02454993117089365:-64>>
  // CHECK: "quantfork.qcast"(%[[ARG_1]])
  // CHECK-SAME: -> tensor<?x!quant.uniform<i8:f32, 0.013075299590241675:-64>>
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[-0.83811146, 2.4960899]> : tensor<2xf32>} : (tensor<?xf32>) -> tensor<?xf32>
  %1 = "quantfork.stats"(%arg1) {layerStats = dense<[-0.835039615, 1.000000e+00]> : tensor<2xf32>} : (tensor<?xf32>) -> tensor<?xf32>
  %2 = "stablehlo.concatenate"(%0, %1) {dimension = 0 : i64} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %3 = "quantfork.stats"(%2) {layerStats = dense<[-0.83811146, 2.4960899]> : tensor<2xf32>} : (tensor<?xf32>) -> tensor<?xf32>
  %4 = "quantfork.stats"(%arg2) {layerStats = dense<[-1.5726943, 1.07351148]> : tensor<2xf32>} : (tensor<?xf32>) -> tensor<?xf32>
  %5 = "stablehlo.concatenate"(%4, %1) {dimension = 0 : i64} : (tensor<?xf32>,  tensor<?xf32>) -> tensor<?xf32>
  %6 = "quantfork.stats"(%5) {layerStats = dense<[-1.5726943, 4.6875381]> : tensor<2xf32>} : (tensor<?xf32>) -> tensor<?xf32>
  func.return  %3, %6 : tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @skip_nan_inf_constant
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<?x112x112x64xf32>) -> tensor<?x56x56x64xf32>
func.func @skip_nan_inf_constant(%arg0: tensor<?x112x112x64xf32>) -> tensor<?x56x56x64xf32> {
  // CHECK-DAG: %[[cst0:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32
  // CHECK-DAG: %[[cst1:.*]] = stablehlo.constant dense<0x7FC00000> : tensor<f32>
  // CHECK-DAG: %[[cst2:.*]] = stablehlo.constant dense<6.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[cst3:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NOT: %[[q0:.*]] = "quantfork.qcast"(%[[cst0]])
  // CHECK-NOT: %[[q1:.*]] = "quantfork.qcast"(%[[cst1]])
  // CHECK: %[[q2:.*]] = "quantfork.qcast"(%[[cst2]])
  // CHECK-SAME: quant.uniform<i8:f32, 0.023529411764705882:-128>
  // CHECK: %[[dq2:.*]] = "quantfork.dcast"(%[[q2]])
  // CHECK-SAME: quant.uniform<i8:f32, 0.023529411764705882:-128>
  // CHECK: %[[q3:.*]] = "quantfork.qcast"(%[[cst3]])
  // CHECK-SAME: quant.uniform<i8:f32, 3.9215686274509805E-9>
  // CHECK: %[[dq3:.*]] = "quantfork.dcast"(%[[q3]])
  // CHECK-SAME: quant.uniform<i8:f32, 3.9215686274509805E-9>
  %0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %1 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
  %2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
  %3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %4 = "stablehlo.add"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %5 = stablehlo.clamp %3, %arg0, %2 : (tensor<f32>, tensor<?x112x112x64xf32>, tensor<f32>) -> tensor<?x112x112x64xf32>
  %6 = "stablehlo.reduce_window"(%5, %4) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %7 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
    stablehlo.return %7 : tensor<f32>
  }) {padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>} : (tensor<?x112x112x64xf32>, tensor<f32>) -> tensor<?x56x56x64xf32>
  return %6 : tensor<?x56x56x64xf32>
}
