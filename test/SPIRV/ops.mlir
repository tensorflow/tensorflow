// RUN: mlir-opt -split-input-file -verify %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.FMul
//===----------------------------------------------------------------------===//

func @fmul_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FMul
  %0 = spv.FMul %arg, %arg : f32
  return %0 : f32
}

func @fmul_vector(%arg: vector<4xf32>) -> vector<4xf32> {
  // CHECK: spv.FMul
  %0 = spv.FMul %arg, %arg : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

func @fmul_i32(%arg: i32) -> i32 {
  // expected-error @+1 {{must be scalar/vector of 16/32/64-bit float}}
  %0 = spv.FMul %arg, %arg : i32
  return %0 : i32
}

// -----

func @fmul_bf16(%arg: bf16) -> bf16 {
  // expected-error @+1 {{must be scalar/vector of 16/32/64-bit float}}
  %0 = spv.FMul %arg, %arg : bf16
  return %0 : bf16
}

// -----

func @fmul_tensor(%arg: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{must be scalar/vector of 16/32/64-bit float}}
  %0 = spv.FMul %arg, %arg : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spv.Return
//===----------------------------------------------------------------------===//

func @return_not_in_func() -> () {
  // expected-error @+1 {{must appear in a 'func' op}}
  spv.Return
}

// -----

func @return_mismatch_func_signature() -> () {
  spv.module {
    func @work() -> (i32) {
      // expected-error @+1 {{cannot be used in functions returning value}}
      spv.Return
    }
  } attributes {
    addressing_model: "Logical",
    memory_model: "VulkanKHR"
  }
  return
}
