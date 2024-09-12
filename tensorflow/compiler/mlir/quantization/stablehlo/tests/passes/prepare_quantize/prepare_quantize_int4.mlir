// RUN: stablehlo-quant-opt %s -split-input-file -stablehlo-prepare-quantize=bit-width=4 -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @dot_int4
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<?x3xf32>) -> tensor<?x2xf32>
func.func @dot_int4(%arg0: tensor<?x3xf32>) -> tensor<?x2xf32> {
  // CHECK: %[[cst:.*]] = stablehlo.constant
  // CHECK: %[[q1:.*]] = "quantfork.qcast"(%[[cst]])
  // CHECK-SAME: quant.uniform<i8:f32, 0.0040316890267764818:127>
  // CHECK: %[[dq1:.*]] = "quantfork.dcast"(%[[q1]])
  // CHECK-SAME: quant.uniform<i8:f32, 0.0040316890267764818:127>
  %cst = stablehlo.constant dense<[[-0.960978984, -0.390246302], [-0.790828585, -0.601039409], [-1.0280807, -1.02731466]]> : tensor<3x2xf32>
  // CHECK: %[[q2:.*]] = "quantfork.qcast"(%[[ARG_0]])
  // CHECK-SAME: quant.uniform<i4:f32, 0.13329366842905679:-1>
  // CHECK: %[[dq2:.*]] = "quantfork.dcast"(%[[q2]])
  // CHECK-SAME: quant.uniform<i4:f32, 0.13329366842905679:-1>
  %0 = "quantfork.stats"(%arg0) {bitsNum = 8 : i64, layerStats = dense<[-0.999415695, 0.99998933]> : tensor<2xf32>, narrowRange = false} : (tensor<?x3xf32>) -> tensor<?x3xf32>
  // CHECK: %[[dot:.*]] = stablehlo.dot %[[dq2]], %[[dq1]]
  %1 = stablehlo.dot %0, %cst : (tensor<?x3xf32>, tensor<3x2xf32>) -> tensor<?x2xf32>
  // CHECK: %[[q3:.*]] = "quantfork.qcast"(%[[dot]])
  // CHECK-SAME: quant.uniform<i4:f32, 0.61633392969767253:-2>>
  // CHECK: %[[dq3:.*]] = "quantfork.dcast"(%[[q3]])
  // CHECK-SAME: quant.uniform<i4:f32, 0.61633392969767253:-2>>
  %2 = "quantfork.stats"(%1) {bitsNum = 8 : i64, layerStats = dense<[-3.6289506, 5.61605835]> : tensor<2xf32>, narrowRange = false} : (tensor<?x2xf32>) -> tensor<?x2xf32>
  // CHECK: return %[[dq3]]
  func.return %2 : tensor<?x2xf32>
}
