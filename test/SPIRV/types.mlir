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

// expected-error @+1 {{spv.array delimiter <...> mismatch}}
func @missing_left_angle_bracket(!spv.array 4xf32>) -> ()

// -----

// expected-error @+1 {{expected array element count followed by 'x' but found 'f32'}}
func @missing_count(!spv.array<f32>) -> ()

// -----

// expected-error @+1 {{expected array element count followed by 'x' but found 'f32'}}
func @missing_x(!spv.array<4 f32>) -> ()

// -----

// expected-error @+1 {{cannot parse type: blabla}}
func @cannot_parse_type(!spv.array<4xblabla>) -> ()

// -----

// expected-error @+1 {{cannot parse type: 3xf32}}
func @more_than_one_dim(!spv.array<4x3xf32>) -> ()

// -----

// expected-error @+1 {{only 1-D vector allowed but found 'vector<4x3xf32>'}}
func @non_1D_vector(!spv.array<4xvector<4x3xf32>>) -> ()

// -----

// expected-error @+1 {{cannot use 'tensor<4xf32>' to compose SPIR-V types}}
func @tensor_type(!spv.array<4xtensor<4xf32>>) -> ()

// -----

// expected-error @+1 {{cannot use 'bf16' to compose SPIR-V types}}
func @bf16_type(!spv.array<4xbf16>) -> ()

// -----

// expected-error @+1 {{only 8/16/32/64-bit integer type allowed but found 'i256'}}
func @i256_type(!spv.array<4xi256>) -> ()

// -----

// expected-error @+1 {{cannot use 'index' to compose SPIR-V types}}
func @index_type(!spv.array<4xindex>) -> ()

// -----

// expected-error @+1 {{cannot use '!llvm.i32' to compose SPIR-V types}}
func @llvm_type(!spv.array<4x!llvm.i32>) -> ()

// -----

//===----------------------------------------------------------------------===//
// RuntimeArrayType
//===----------------------------------------------------------------------===//

// CHECK: func @scalar_runtime_array_type(!spv.rtarray<f32>, !spv.rtarray<i32>)
func @scalar_runtime_array_type(!spv.rtarray<f32>, !spv.rtarray<i32>) -> ()

// CHECK: func @vector_runtime_array_type(!spv.rtarray<vector<4xf32>>)
func @vector_runtime_array_type(!spv.rtarray< vector<4xf32> >) -> ()

// -----

// expected-error @+1 {{spv.rtarray delimiter <...> mismatch}}
func @missing_left_angle_bracket(!spv.rtarray f32>) -> ()

// -----

// expected-error @+1 {{cannot parse type: 4xf32}}
func @redundant_count(!spv.rtarray<4xf32>) -> ()
