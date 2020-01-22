// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main(%arg0 : tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
  %result = "xla_hlo.reverse"(%arg0) {
    dimensions = dense<[1,2]> : tensor<2xi64>
  } : (tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32>
  return %result : tensor<10x11x12x13xf32>
}

// CHECK-LABEL: main
// CHECK: %[[ARG0:.*]] = f32[10,11,12,13] parameter(0)
// CHECK: ROOT %[[RESULT:.*]] = f32[10,11,12,13] reverse(f32[10,11,12,13] %[[ARG0]]), dimensions={1,2}
