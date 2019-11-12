// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main() -> tensor<2x3x5xf32> {
  %0 = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = xla_hlo.constant dense<1.000000e+00> : tensor<f32>
  %2 = xla_hlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  %3 = "xla_hlo.rng_uniform"(%0, %1, %2) : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  return %3 : tensor<2x3x5xf32>
}

// CHECK: ENTRY %main
// CHECK-DAG: %[[A:.*]] = f32[] constant(0)
// CHECK-DAG: %[[B:.*]] = f32[] constant(1)
// CHECK: ROOT %[[RESULT:.*]] = f32[2,3,5] rng(f32[] %[[A]], f32[] %[[B]]), distribution=rng_uniform
