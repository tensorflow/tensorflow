// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

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

// -----

//===----------------------------------------------------------------------===//
// spv.Variable
//===----------------------------------------------------------------------===//

func @variable_no_init(%arg0: f32) -> () {
  // CHECK: spv.Variable : !spv.ptr<f32, Function>
  %0 = spv.Variable : !spv.ptr<f32, Function>
  return
}

func @variable_init() -> () {
  %0 = spv.constant 4.0 : f32
  // CHECK: spv.Variable init(%0) : !spv.ptr<f32, Private>
  %1 = spv.Variable init(%0) : !spv.ptr<f32, Private>
  return
}

func @variable_bind() -> () {
  // CHECK: spv.Variable bind(1, 2) : !spv.ptr<f32, Uniform>
  %0 = spv.Variable bind(1, 2) : !spv.ptr<f32, Uniform>
  return
}

func @variable_init_bind() -> () {
  %0 = spv.constant 4.0 : f32
  // CHECK: spv.Variable init(%0) {binding: 5 : i32} : !spv.ptr<f32, Private>
  %1 = spv.Variable init(%0) {binding: 5 : i32} : !spv.ptr<f32, Private>
  return
}

// -----

func @expect_ptr_result_type(%arg0: f32) -> () {
  // expected-error @+1 {{expected spv.ptr type}}
  %0 = spv.Variable : f32
  return
}

// -----

func @variable_init(%arg0: f32) -> () {
  // expected-error @+1 {{op initializer must be the result of a spv.Constant or module-level spv.Variable op}}
  %0 = spv.Variable init(%arg0) : !spv.ptr<f32, Private>
  return
}

// -----

func @storage_class_mismatch() -> () {
  %0 = spv.constant 5.0 : f32
  // expected-error @+1 {{storage class must match result pointer's storage class}}
  %1 = "spv.Variable"(%0) {storage_class : "Uniform"} : (f32) -> !spv.ptr<f32, Function>
  return
}

// -----

func @cannot_be_generic_storage_class(%arg0: f32) -> () {
  // expected-error @+1 {{storage class cannot be 'Generic'}}
  %0 = spv.Variable : !spv.ptr<f32, Generic>
  return
}
