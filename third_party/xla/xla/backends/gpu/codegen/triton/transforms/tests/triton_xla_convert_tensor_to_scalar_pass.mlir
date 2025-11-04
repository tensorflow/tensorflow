// RUN: xla-opt %s -split-input-file -triton-xla-convert-0d-tensor-to-scalar \
// RUN: | FileCheck %s

func.func @addf(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: arith.addf {{.*}} : f32
  %0 = arith.addf %arg0, %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @addf() -> tensor<f32> {
  // CHECK: arith.constant 1.000000e+00 : f32
  %0 = arith.constant dense<1.0> : tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @addf(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: tt.extern_elementwise {{.*}}{libname = "dev", libpath = "/path",
  // CHECK-SAME: pure = true, symbol = "sym"} : (f32) -> f32
  %0 = tt.extern_elementwise %arg0
    {libname = "dev",
     libpath = "/path",
     pure = true,
     symbol = "sym"} : (tensor<f32>) -> tensor<f32> 
  return %0 : tensor<f32>
}
