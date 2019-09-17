// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.GLSL.Exp
//===----------------------------------------------------------------------===//

func @exp(%arg0 : f32) -> () {
  // CHECK: spv.GLSL.Exp {{%.*}} : f32
  %2 = spv.GLSL.Exp %arg0 : f32
  return
}

func @expvec(%arg0 : vector<3xf16>) -> () {
  // CHECK: spv.GLSL.Exp {{%.*}} : vector<3xf16>
  %2 = spv.GLSL.Exp %arg0 : vector<3xf16>
  return
}

// -----

func @exp(%arg0 : i32) -> () {
  // expected-error @+1 {{op operand #0 must be 16/32-bit float or vector of 16/32-bit float values}}
  %2 = spv.GLSL.Exp %arg0 : i32
  return
}

// -----

func @exp(%arg0 : vector<5xf32>) -> () {
  // expected-error @+1 {{op operand #0 must be 16/32-bit float or vector of 16/32-bit float values of length 2/3/4}}
  %2 = spv.GLSL.Exp %arg0 : vector<5xf32>
  return
}

// -----

func @exp(%arg0 : f32, %arg1 : f32) -> () {
  // expected-error @+1 {{expected ':'}}
  %2 = spv.GLSL.Exp %arg0, %arg1 : i32
  return
}

// -----

func @exp(%arg0 : i32) -> () {
  // expected-error @+2 {{expected non-function type}}
  %2 = spv.GLSL.Exp %arg0 :
  return
}
