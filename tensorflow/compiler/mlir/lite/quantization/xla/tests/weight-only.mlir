// RUN: tf-opt -xla-hlo-propagate-quant %s | FileCheck %s

// CHECK-LABEL: func @mul
func @mul(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[w:.*]] = constant dense<{{\[\[}}-1.000000e+00, -5.000000e-01], [5.000000e-01, 1.000000e+00]]> : tensor<2x2xf32>
// CHECK-NEXT: %[[q:.*]] = "quant.qcast"(%[[w]]) : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 0.0078431372549019607:128>>
// CHECK-NEXT: %[[dq:.*]] = "quant.dcast"(%[[q]]) : (tensor<2x2x!quant.uniform<u8:f32, 0.0078431372549019607:128>>) -> tensor<2x2xf32>
// CHECK-NEXT: %[[mul:.*]] = xla_hlo.multiply %arg0, %[[dq]] : tensor<2x2xf32>
// CHECK-NEXT: return %[[mul]] : tensor<2x2xf32>
  %w = constant dense<[[-1.0, -0.5], [0.5, 1.0]]> : tensor<2x2xf32>
  %mul = xla_hlo.multiply %arg0, %w : tensor<2x2xf32>
  return %mul: tensor<2x2xf32>
}

// CHECK-LABEL: func @add
func @add(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK: %[[b:.*]] = constant dense<1.000000e+00> : tensor<2xf32>
// CHECK-NEXT: %[[q:.*]] = "quant.qcast"(%[[b]]) : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 0.0039215686274509803>>
// CHECK-NEXT: %[[dq:.*]] = "quant.dcast"(%[[q]]) : (tensor<2x!quant.uniform<u8:f32, 0.0039215686274509803>>) -> tensor<2xf32>
// CHECK-NEXT: %[[add:.*]] = "xla_hlo.add"(%arg0, %[[dq]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<2x2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT: return %[[add]] : tensor<2x2xf32>
  %b = constant dense<1.0> : tensor<2xf32>
  %add = "xla_hlo.add"(%arg0, %b) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<2x2xf32>, tensor<2xf32>) -> tensor<2x2xf32>
  return %add: tensor<2x2xf32>
}
