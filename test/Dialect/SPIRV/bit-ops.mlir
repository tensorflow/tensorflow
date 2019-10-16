// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.BitwiseOr
//===----------------------------------------------------------------------===//

func @bitwise_or_scalar(%arg: i32) -> i32 {
  // CHECK: spv.BitwiseOr
  %0 = spv.BitwiseOr %arg, %arg : i32
  return %0 : i32
}

func @bitwise_or_vector(%arg: vector<4xi32>) -> vector<4xi32> {
  // CHECK: spv.BitwiseOr
  %0 = spv.BitwiseOr %arg, %arg : vector<4xi32>
  return %0 : vector<4xi32>
}

// -----

func @bitwise_or_float(%arg0: f16, %arg1: f16) -> f16 {
  // expected-error @+1 {{operand #0 must be 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4}}
  %0 = spv.BitwiseOr %arg0, %arg1 : f16
  return %0 : f16
}

// -----

//===----------------------------------------------------------------------===//
// spv.BitwiseXor
//===----------------------------------------------------------------------===//

func @bitwise_xor_scalar(%arg: i32) -> i32 {
  // CHECK: spv.BitwiseXor
  %0 = spv.BitwiseXor %arg, %arg : i32
  return %0 : i32
}

func @bitwise_xor_vector(%arg: vector<4xi32>) -> vector<4xi32> {
  // CHECK: spv.BitwiseXor
  %0 = spv.BitwiseXor %arg, %arg : vector<4xi32>
  return %0 : vector<4xi32>
}

// -----

func @bitwise_xor_float(%arg0: f16, %arg1: f16) -> f16 {
  // expected-error @+1 {{operand #0 must be 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4}}
  %0 = spv.BitwiseXor %arg0, %arg1 : f16
  return %0 : f16
}

// -----

//===----------------------------------------------------------------------===//
// spv.BitwiseAnd
//===----------------------------------------------------------------------===//

func @bitwise_and_scalar(%arg: i32) -> i32 {
  // CHECK: spv.BitwiseAnd
  %0 = spv.BitwiseAnd %arg, %arg : i32
  return %0 : i32
}

func @bitwise_and_vector(%arg: vector<4xi32>) -> vector<4xi32> {
  // CHECK: spv.BitwiseAnd
  %0 = spv.BitwiseAnd %arg, %arg : vector<4xi32>
  return %0 : vector<4xi32>
}

// -----

func @bitwise_and_float(%arg0: f16, %arg1: f16) -> f16 {
  // expected-error @+1 {{operand #0 must be 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4}}
  %0 = spv.BitwiseAnd %arg0, %arg1 : f16
  return %0 : f16
}
