// RUN: stablehlo-quant-opt %s -split-input-file -stablehlo-prepare-srq-quantize | FileCheck %s

func.func @main(%arg0: tensor<?x3xf32>) -> tensor<?x2xf32> {
  %cst = stablehlo.constant dense<[[-0.960978984, -0.390246302], [-0.790828585, -0.601039409], [-1.0280807, -1.02731466]]> : tensor<3x2xf32>
  %0 = "quantfork.stats"(%arg0) {bitsNum = 8 : i64, layerStats = dense<[-0.999415695, 0.99998933]> : tensor<2xf32>, narrowRange = false} : (tensor<?x3xf32>) -> tensor<?x3xf32>
  %1 = stablehlo.dot %0, %cst : (tensor<?x3xf32>, tensor<3x2xf32>) -> tensor<?x2xf32>
  %2 = "quantfork.stats"(%1) {bitsNum = 8 : i64, layerStats = dense<[-3.6289506, 5.61605835]> : tensor<2xf32>, narrowRange = false} : (tensor<?x2xf32>) -> tensor<?x2xf32>
  func.return %2 : tensor<?x2xf32>
}

// CHECK: %[[cst:.*]] = arith.constant
// CHECK: %[[q1:.*]] = "quantfork.qcast"(%arg0)
// CHECK-SAME: quant.uniform<i8:f32, 0.0078408040252386357:-1>
// CHECK: %[[dq1:.*]] = "quantfork.dcast"(%[[q1]])
// CHECK-SAME: quant.uniform<i8:f32, 0.0078408040252386357:-1>
// CHECK: %[[dot:.*]] = stablehlo.dot %[[dq1]], %[[cst]]
// CHECK: %[[q2:.*]] = "quantfork.qcast"(%[[dot]])
// CHECK-SAME: quant.uniform<i8:f32, 0.036254937041039562:-28>>
// CHECK: %[[dq2:.*]] = "quantfork.dcast"(%[[q2]])
// CHECK-SAME: quant.uniform<i8:f32, 0.036254937041039562:-28>>
// CHECK: return %[[dq2]]
