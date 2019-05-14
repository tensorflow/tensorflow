// RUN: mlir-opt %s -split-input-file -quant-convert-const | FileCheck %s --dump-input=fail

// Magic numbers:
//   7.8125e-03 = 1/128 = 2/256 : real range = [-1.0, 0.9921875] (for 8bit, zeroPoint=128)
//   1.250000e-01 = 1/8 = 2/16  : real range = [-1.0, 0.875] (for 4bit, zeroPoint=8)

// -----
// Verifies u8 affine quantization on a splat tensor.
// Note that MLIR prints int attributes as signed, so the constant, when
// quantized, is the signed printed version of an unsigned quantity
// (-64 signed == 192 unsigned).
// CHECK-LABEL: constant_splat_tensor_u8_affine
func @constant_splat_tensor_u8_affine() -> tensor<4xf32> {
  // CHECK: %cst = constant splat<tensor<4xi8>, -64>
  // CHECK-NEXT: %0 = "quant.scast"(%cst) : (tensor<4xi8>) -> tensor<4x!quant.uniform<u8:f32, 7.812500e-03:128>>
  %cst = constant splat<tensor<4xf32>, 0.5>
  %1 = "quant.qcast"(%cst) : (tensor<4xf32>) -> tensor<4x!quant.uniform<u8:f32, 7.812500e-03:128>>
  %2 = "quant.dcast"(%1) : (tensor<4x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> (tensor<4xf32>)
  return %2 : tensor<4xf32>
}

// -----
// Verifies i8 affine quantization on a splat tensor.
// CHECK-LABEL: constant_splat_tensor_i8_affine
func @constant_splat_tensor_i8_affine() -> tensor<4xf32> {
  // CHECK: %cst = constant splat<tensor<4xi8>, 63>
  // CHECK-NEXT: %0 = "quant.scast"(%cst) : (tensor<4xi8>) -> tensor<4x!quant.uniform<i8:f32, 7.812500e-03:-1>>
  %cst = constant splat<tensor<4xf32>, 0.5>
  %1 = "quant.qcast"(%cst) : (tensor<4xf32>) -> tensor<4x!quant.uniform<i8:f32, 7.812500e-03:-1>>
  %2 = "quant.dcast"(%1) : (tensor<4x!quant.uniform<i8:f32, 7.812500e-03:-1>>) -> (tensor<4xf32>)
  return %2 : tensor<4xf32>
}

// -----
// Verifies i8 fixedpoint quantization on a splat tensor.
// CHECK-LABEL: const_splat_tensor_i8_fixedpoint
func @const_splat_tensor_i8_fixedpoint() -> tensor<4xf32> {
  // CHECK: %cst = constant splat<tensor<4xi8>, 64>
  // CHECK-NEXT: %0 = "quant.scast"(%cst) : (tensor<4xi8>) -> tensor<4x!quant.uniform<i8:f32, 7.812500e-03>>
  %cst = constant splat<tensor<4xf32>, 0.5>
  %1 = "quant.qcast"(%cst) : (tensor<4xf32>) -> tensor<4x!quant.uniform<i8:f32, 7.812500e-03>>
  %2 = "quant.dcast"(%1) : (tensor<4x!quant.uniform<i8:f32, 7.812500e-03>>) -> (tensor<4xf32>)
  return %2 : tensor<4xf32>
}

// -----
// Verifies i8 fixedpoint quantization on a splat tensor resulting in a negative storage value.
// CHECK-LABEL: const_splat_tensor_i8_fixedpoint_neg
func @const_splat_tensor_i8_fixedpoint_neg() -> tensor<4xf32> {
  // CHECK: %cst = constant splat<tensor<4xi8>, -64>
  %cst = constant splat<tensor<4xf32>, -0.5>
  %1 = "quant.qcast"(%cst) : (tensor<4xf32>) -> tensor<4x!quant.uniform<i8:f32, 7.812500e-03>>
  %2 = "quant.dcast"(%1) : (tensor<4x!quant.uniform<i8:f32, 7.812500e-03>>) -> (tensor<4xf32>)
  return %2 : tensor<4xf32>
}

// -----
// Verifies i8 fixedpoint quantization on a dense tensor, sweeping values.
// CHECK-LABEL: const_dense_tensor_i8_fixedpoint
func @const_dense_tensor_i8_fixedpoint() -> tensor<7xf32> {
  // CHECK: %cst = constant dense<tensor<7xi8>, [-128, -128, -64, 0, 64, 127, 127]>
  %cst = constant dense<tensor<7xf32>, [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]>
  %1 = "quant.qcast"(%cst) : (tensor<7xf32>) -> tensor<7x!quant.uniform<i8:f32, 7.812500e-03>>
  %2 = "quant.dcast"(%1) : (tensor<7x!quant.uniform<i8:f32, 7.812500e-03>>) -> (tensor<7xf32>)
  return %2 : tensor<7xf32>
}

// -----
// Verifies i8 fixedpoint quantization on a sparse tensor, sweeping values.
// CHECK-LABEL: const_sparse_tensor_i8_fixedpoint
func @const_sparse_tensor_i8_fixedpoint() -> tensor<7x2xf32> {
  // NOTE: Ugly regex match pattern for opening "[[" of indices tensor.
  // CHECK: %cst = constant sparse<tensor<7x2xi8>, {{\[}}[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]], [-128, -128, -64, 0, 64, 127, 127]>
  %cst = constant sparse<tensor<7x2xf32>,
      [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]],
      [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]>
  %1 = "quant.qcast"(%cst) : (tensor<7x2xf32>) -> tensor<7x2x!quant.uniform<i8:f32, 7.812500e-03>>
  %2 = "quant.dcast"(%1) : (tensor<7x2x!quant.uniform<i8:f32, 7.812500e-03>>) -> (tensor<7x2xf32>)
  return %2 : tensor<7x2xf32>
}

// -----
// Verifies i8 fixedpoint quantization on a primitive const.
// CHECK-LABEL: const_primitive_float_i8_fixedpoint
func @const_primitive_float_i8_fixedpoint() -> f32 {
  // CHECK: %c64_i8 = constant 64 : i8
  // CHECK-NEXT: %0 = "quant.scast"(%c64_i8) : (i8) -> !quant.uniform<i8:f32, 7.812500e-03>
  %cst = constant 0.5 : f32
  %1 = "quant.qcast"(%cst) : (f32) -> !quant.uniform<i8:f32, 7.812500e-03>
  %2 = "quant.dcast"(%1) : (!quant.uniform<i8:f32, 7.812500e-03>) -> (f32)
  return %2 : f32
}

// -----
// Verifies u4 affine quantization on a dense tensor, sweeping values.
// CHECK-LABEL: const_dense_tensor_u4_affine
func @const_dense_tensor_u4_affine() -> tensor<7xf32> {
  // NOTE: Unsigned quantities printed by MLIR as signed.
  // CHECK: %cst = constant dense<tensor<7xi4>, [0, 0, 4, -8, -4, -1, -1]>
  %cst = constant dense<tensor<7xf32>, [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]>
  %1 = "quant.qcast"(%cst) : (tensor<7xf32>) -> tensor<7x!quant.uniform<u4:f32, 1.250000e-01:8>>
  %2 = "quant.dcast"(%1) : (tensor<7x!quant.uniform<u4:f32, 1.250000e-01:8>>) -> (tensor<7xf32>)
  return %2 : tensor<7xf32>
}

// -----
// Verifies i4 affine quantization on a dense tensor, sweeping values.
// CHECK-LABEL: const_dense_tensor_i4_affine
func @const_dense_tensor_i4_affine() -> tensor<7xf32> {
  // NOTE: Unsigned quantities printed by MLIR as signed.
  // CHECK: %cst = constant dense<tensor<7xi4>, [-8, -8, -5, -1, 3, 7, 7]>
  %cst = constant dense<tensor<7xf32>, [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]>
  %1 = "quant.qcast"(%cst) : (tensor<7xf32>) -> tensor<7x!quant.uniform<i4:f32, 1.250000e-01:-1>>
  %2 = "quant.dcast"(%1) : (tensor<7x!quant.uniform<i4:f32, 1.250000e-01:-1>>) -> (tensor<7xf32>)
  return %2 : tensor<7xf32>
}

// -----
// Verifies i4 fixed point quantization on a dense tensor, sweeping values.
// CHECK-LABEL: const_dense_tensor_i4_fixedpoint
func @const_dense_tensor_i4_fixedpoint() -> tensor<7xf32> {
  // CHECK: %cst = constant dense<tensor<7xi4>, [-8, -8, -4, 0, 4, 7, 7]>
  %cst = constant dense<tensor<7xf32>, [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]>
  %1 = "quant.qcast"(%cst) : (tensor<7xf32>) -> tensor<7x!quant.uniform<i4:f32, 1.250000e-01>>
  %2 = "quant.dcast"(%1) : (tensor<7x!quant.uniform<i4:f32, 1.250000e-01>>) -> (tensor<7xf32>)
  return %2 : tensor<7xf32>
}

// -----
// Verifies i8 fixedpoint quantization on a dense tensor, sweeping values, and
// custom storage range. (the -128 should be clamped to -100, and the 127 should
// be clamped to 100).
// CHECK-LABEL: const_custom_storage_range_i8_fixedpoint
func @const_custom_storage_range_i8_fixedpoint() -> tensor<7xf32> {
  // CHECK: %cst = constant dense<tensor<7xi8>, [-100, -100, -64, 0, 64, 100, 100]>
  %cst = constant dense<tensor<7xf32>, [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]>
  %1 = "quant.qcast"(%cst) : (tensor<7xf32>) -> tensor<7x!quant.uniform<i8<-100:100>:f32, 7.812500e-03>>
  %2 = "quant.dcast"(%1) : (tensor<7x!quant.uniform<i8<-100:100>:f32, 7.812500e-03>>) -> (tensor<7xf32>)
  return %2 : tensor<7xf32>
}
