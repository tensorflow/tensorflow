// RUN: emitters_opt %s \
// RUN: -split-input-file -convert-elementwise-0d-tensor-to-scalar \
// RUN: | FileCheck %s

func.func @converts_0d_addf(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: arith.addf {{.*}} : f32
  %0 = arith.addf %arg0, %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @skips_1d_addf(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK: arith.addf {{.*}} : tensor<1xf32>
  %0 = arith.addf %arg0, %arg0 : tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

func.func @converts_0d_constant() -> tensor<f32> {
  // CHECK: arith.constant 1.000000e+00 : f32
  %0 = arith.constant dense<1.0> : tensor<f32>
  return %0 : tensor<f32>
}
