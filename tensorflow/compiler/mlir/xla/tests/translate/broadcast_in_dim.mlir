// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main(%arg0: tensor<1xf32>) -> tensor<1x10xf32> {
  %result = "xla_hlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = dense<0> : tensor<1xi64>
  } : (tensor<1xf32>) -> tensor<1x10xf32>
  return %result : tensor<1x10xf32>
}

// CHECK: ENTRY %main.3 ([[ARG0:.*]]: f32[1]) -> f32[1,10] {
// CHECK:  %[[ARG0]] = f32[1] parameter(0)
// CHECK:  ROOT %broadcast.2 = f32[1,10] broadcast(f32[1] %[[ARG0]]), dimensions={0}
