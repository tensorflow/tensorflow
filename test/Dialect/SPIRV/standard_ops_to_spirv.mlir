// RUN: mlir-opt -convert-std-to-spirv %s -o - | FileCheck %s

// CHECK-LABEL: @fmul_scalar
func @fmul_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FMul
  %0 = mulf %arg, %arg : f32
  return %0 : f32
}

// CHECK-LABEL: @fmul_vector2
func @fmul_vector2(%arg: vector<2xf32>) -> vector<2xf32> {
  // CHECK: spv.FMul
  %0 = mulf %arg, %arg : vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: @fmul_vector3
func @fmul_vector3(%arg: vector<3xf32>) -> vector<3xf32> {
  // CHECK: spv.FMul
  %0 = mulf %arg, %arg : vector<3xf32>
  return %0 : vector<3xf32>
}

// CHECK-LABEL: @fmul_vector4
func @fmul_vector4(%arg: vector<4xf32>) -> vector<4xf32> {
  // CHECK: spv.FMul
  %0 = mulf %arg, %arg : vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: @fmul_vector5
func @fmul_vector5(%arg: vector<5xf32>) -> vector<5xf32> {
  // Vector length of only 2, 3, and 4 is valid for SPIR-V
  // CHECK: mulf
  %0 = mulf %arg, %arg : vector<5xf32>
  return %0 : vector<5xf32>
}

// CHECK-LABEL: @fmul_tensor
func @fmul_tensor(%arg: tensor<4xf32>) -> tensor<4xf32> {
  // For tensors mulf cannot be lowered directly to spv.FMul
  // CHECK: mulf
  %0 = mulf %arg, %arg : tensor<4xf32>
  return %0 : tensor<4xf32>
}

