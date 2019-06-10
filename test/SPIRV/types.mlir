// RUN: mlir-opt -split-input-file -verify %s | FileCheck %s

// TODO(b/133530217): Add more tests after switching to the generic parser.

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

// CHECK: func @scalar_array_type(!spv.array<16 x f32>, !spv.array<8 x i32>)
func @scalar_array_type(!spv.array<16xf32>, !spv.array<8 x i32>) -> ()

// CHECK: func @vector_array_type(!spv.array<32 x vector<4xf32>>)
func @vector_array_type(!spv.array< 32 x vector<4xf32> >) -> ()

// -----

// expected-error @+1 {{unknown SPIR-V type}}
func @missing_count(!spv.array<f32>) -> ()

// -----

// expected-error @+1 {{unknown SPIR-V type}}
func @missing_x(!spv.array<4 f32>) -> ()

// -----

// expected-error @+1 {{unknown SPIR-V type}}
func @more_than_one_dim(!spv.array<4x3xf32>) -> ()

// -----

//===----------------------------------------------------------------------===//
// RuntimeArrayType
//===----------------------------------------------------------------------===//

// CHECK: func @scalar_runtime_array_type(!spv.rtarray<f32>, !spv.rtarray<i32>)
func @scalar_runtime_array_type(!spv.rtarray<f32>, !spv.rtarray<i32>) -> ()

// CHECK: func @vector_runtime_array_type(!spv.rtarray<vector<4xf32>>)
func @vector_runtime_array_type(!spv.rtarray< vector<4xf32> >) -> ()

// -----

// expected-error @+1 {{unknown SPIR-V type}}
func @redundant_count(!spv.rtarray<4xf32>) -> ()
