// RUN: tf-opt -xla-hlo-materialize-quant %s | FileCheck %s

// CHECK-LABEL: func @quantize_rewrite
func @quantize_rewrite(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
// CHECK:      %[[qcst:.*]] = constant dense<{{\[\[}}21004416], [-1056997248]]> : tensor<2x1xi32>
// CHECK-NEXT: %[[dq:.*]] = "xla_hlo.dequantize"(%[[qcst]]) {is_16bits = false, max_range = 0.996078431 : f32, min_range = -1.00392163 : f32,
// CHECK-SAME:   mode = "MIN_COMBINED", transpose_output = false} : (tensor<2x1xi32>) -> tensor<2x4xbf16>
// CHECK-NEXT: %[[cast:.*]] = "xla_hlo.convert"(%[[dq]]) : (tensor<2x4xbf16>) -> tensor<2x4xf32>
// CHECK-NEXT: %[[mul:.*]] = xla_hlo.mul %arg0, %[[cast]] : tensor<2x4xf32>
// CHECK-NEXT: return %[[mul]] : tensor<2x4xf32>

  %w = constant dense<[[-1.0, -0.5, 0.0, 0.0], [0.5, 1.0, 0.0, 0.0]]> : tensor<2x4xf32>
  %q = "quant.qcast"(%w) : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<u8:f32, 0.0078431372549019607:128>>
  %dq = "quant.dcast"(%q) : (tensor<2x4x!quant.uniform<u8:f32, 0.0078431372549019607:128>>) -> tensor<2x4xf32>
  %mul = xla_hlo.mul %arg0, %dq : tensor<2x4xf32>
  return %mul: tensor<2x4xf32>
}

// CHECK-LABEL: func @quantize_small
func @quantize_small(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
// CHECK: %[[w:.*]] = constant dense<1.000000e+00> : tensor<1x4xf32>
// CHECK-NEXT: %[[mul:.*]] = xla_hlo.mul %arg0, %[[w]] : tensor<1x4xf32>
// CHECK-NEXT: return %[[mul]] : tensor<1x4xf32>

  %w = constant dense<1.0> : tensor<1x4xf32>
  %q = "quant.qcast"(%w) : (tensor<1x4xf32>) -> tensor<1x4x!quant.uniform<u8:f32, 0.0078431372549019607:128>>
  %dq = "quant.dcast"(%q) : (tensor<1x4x!quant.uniform<u8:f32, 0.0078431372549019607:128>>) -> tensor<1x4xf32>
  %mul = xla_hlo.mul %arg0, %dq : tensor<1x4xf32>
  return %mul: tensor<1x4xf32>
}

// CHECK-LABEL: func @quantize_non_cst
func @quantize_non_cst(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
// CHECK-NEXT: %[[mul:.*]] = xla_hlo.mul %arg0, %arg0 : tensor<2x4xf32>
// CHECK-NEXT: return %[[mul]] : tensor<2x4xf32>

  %q = "quant.qcast"(%arg0) : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<u8:f32, 0.0078431372549019607:128>>
  %dq = "quant.dcast"(%q) : (tensor<2x4x!quant.uniform<u8:f32, 0.0078431372549019607:128>>) -> tensor<2x4xf32>
  %mul = xla_hlo.mul %arg0, %dq : tensor<2x4xf32>
  return %mul: tensor<2x4xf32>
}

// CHECK-LABEL: func @quantize_non_4x
func @quantize_non_4x(%arg0: tensor<2x5xf32>) -> tensor<2x5xf32> {
// CHECK: %[[w:.*]] = constant dense<1.000000e+00> : tensor<2x5xf32>
// CHECK-NEXT: %[[mul:.*]] = xla_hlo.mul %arg0, %[[w]] : tensor<2x5xf32>
// CHECK-NEXT: return %[[mul]] : tensor<2x5xf32>

  %w = constant dense<1.0> : tensor<2x5xf32>
  %q = "quant.qcast"(%w) : (tensor<2x5xf32>) -> tensor<2x5x!quant.uniform<u8:f32, 0.0078431372549019607:128>>
  %dq = "quant.dcast"(%q) : (tensor<2x5x!quant.uniform<u8:f32, 0.0078431372549019607:128>>) -> tensor<2x5xf32>
  %mul = xla_hlo.mul %arg0, %dq : tensor<2x5xf32>
  return %mul: tensor<2x5xf32>
}
