// RUN: xla-opt %s -split-input-file -unsupported-elementwise-to-triton \
// RUN: | FileCheck %s

func.func @converts_tensor_negf_to_subf(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<10xf32>
  // CHECK: %[[SUB:.*]] = arith.subf %[[ZERO]], %arg0 : tensor<10xf32>
  %0 = arith.negf %arg0 : tensor<10xf32>
  // CHECK: return %[[SUB]] : tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

//-----

func.func @converts_scalar_negf_to_subf(%arg0: f32) -> f32 {
  // CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SUB:.*]] = arith.subf %[[ZERO]], %arg0 : f32
  %0 = arith.negf %arg0 : f32
  // CHECK: return %[[SUB]] : f32
  func.return %0 : f32
}
