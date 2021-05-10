// RUN: tfjs-opt -split-input-file -verify-diagnostics -tfl-runtime-verify %s | FileCheck %s

// -----

func @testPReluWrongArgumentAndResultTypes(%arg0: tensor<10x10x10x10xf32>, %arg1: tensor<1x1x10xi32>) -> tensor<10x10x10xf32> {
  // expected-error @+1 {{requires the same element type for all operands and results}}
  %0 = tfjs.Prelu %arg0, %arg1 : (tensor<10x10x10x10xf32>, tensor<1x1x10xi32>) -> tensor<10x10x10x10xi32>
  return %0 : tensor<10x10x10x10xi32>
}

// -----

func @testPReluWrongOutputShape(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<2x3x4xf32>) -> tensor<1x2x3x5xf32> {
  // expected-error @+1 {{op result type '1x2x3x5' not broadcast compatible with broadcasted operands's shapes '1x2x3x4'}}
  %0 = tfjs.Prelu %arg0, %arg1 : (tensor<1x2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<1x2x3x5xf32>
  return %0 : tensor<1x2x3x5xf32>
}

// -----

func @testPReluWrongAlphaRank(%arg0: tensor<7x3x2x14xf32>, %arg1: tensor<2x7x3x2x14xf32>) -> tensor<7x3x2x14xf32> {
  // expected-error @+1 {{result type '7x3x2x14' not broadcast compatible with broadcasted operands's shapes '2x7x3x2x14'}}
  %0 = tfjs.Prelu %arg0, %arg1 : (tensor<7x3x2x14xf32>, tensor<2x7x3x2x14xf32>) -> tensor<7x3x2x14xf32>
  return %0 : tensor<7x3x2x14xf32>
}

// -----

func @testPReluInvalidBroadcast(%arg0: tensor<15x14x2x14xf32>, %arg1: tensor<1x1x3xf32>) -> tensor<15x14x2x14xf32> {
  // expected-error @+1 {{op operands don't have broadcast-compatible shapes}}
  %0 = tfjs.Prelu %arg0, %arg1 : (tensor<15x14x2x14xf32>, tensor<1x1x3xf32>) -> tensor<15x14x2x14xf32>
  return %0 : tensor<15x14x2x14xf32>
}
// -----
// CHECK-LABEL: func @testPReluValidSameSize
func @testPReluValidSameSize(%arg0: tensor<16x20x20x13xf32>, %arg1: tensor<20x20x13xf32>) -> tensor<16x20x20x13xf32> {
  %0 = tfjs.Prelu %arg0, %arg1 : (tensor<16x20x20x13xf32>, tensor<20x20x13xf32>) -> tensor<16x20x20x13xf32>
  return %0 : tensor<16x20x20x13xf32>
}

// -----
// CHECK-LABEL: func @testPReluValidBroadcast
func @testPReluValidBroadcast(%arg0: tensor<19x7x12x14xf32>, %arg1: tensor<1x1x14xf32>) -> tensor<19x7x12x14xf32> {
  %0 = tfjs.Prelu %arg0, %arg1 : (tensor<19x7x12x14xf32>, tensor<1x1x14xf32>) -> tensor<19x7x12x14xf32>
  return %0 : tensor<19x7x12x14xf32>
}

// -----
// CHECK-LABEL: func @testPReluValidFullBroadcast
func @testPReluValidFullBroadcast(%arg0: tensor<7x8x9x10xf32>, %arg1: tensor<1x1x1xf32>) -> tensor<7x8x9x10xf32> {
  %0 = tfjs.Prelu %arg0, %arg1 : (tensor<7x8x9x10xf32>, tensor<1x1x1xf32>) -> tensor<7x8x9x10xf32>
  return %0 : tensor<7x8x9x10xf32>
}

