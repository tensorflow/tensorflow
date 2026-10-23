// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: litert-opt %s --tfl-fold-stablehlo-constant-transforms | FileCheck %s

// CHECK-LABEL: func.func @transpose_dense_i8
func.func @transpose_dense_i8() -> tensor<3x2xi8> {
  // CHECK: %[[CST:.+]] = stablehlo.constant dense<{{\[\[}}1, 4], [2, 5], [3, 6]]> : tensor<3x2xi8>
  // CHECK-NOT: stablehlo.transpose
  // CHECK: return %[[CST]]
  %0 = stablehlo.constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi8>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xi8>) -> tensor<3x2xi8>
  return %1 : tensor<3x2xi8>
}

// CHECK-LABEL: func.func @transpose_dense_i4
func.func @transpose_dense_i4() -> tensor<3x2xi4> {
  // CHECK: %[[CST:.+]] = stablehlo.constant dense<{{\[\[}}1, 4], [2, -5], [3, -6]]> : tensor<3x2xi4>
  // CHECK-NOT: stablehlo.transpose
  // CHECK: return %[[CST]]
  %0 = stablehlo.constant dense<[[1, 2, 3], [4, -5, -6]]> : tensor<2x3xi4>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xi4>) -> tensor<3x2xi4>
  return %1 : tensor<3x2xi4>
}

// CHECK-LABEL: func.func @transpose_dense_i2_rank3
func.func @transpose_dense_i2_rank3() -> tensor<3x2x2xi2> {
  // CHECK: %[[CST:.+]] = stablehlo.constant dense<{{\[\[\[}}0, 1], [-2, -1]], {{\[\[}}-2, -1], [0, 1]], {{\[\[}}0, 1], [-2, -1]]]> : tensor<3x2x2xi2>
  // CHECK-NOT: stablehlo.transpose
  // CHECK: return %[[CST]]
  %0 = stablehlo.constant dense<[[[0, 1], [-2, -1], [0, 1]], [[-2, -1], [0, 1], [-2, -1]]]> : tensor<2x3x2xi2>
  %1 = stablehlo.transpose %0, dims = [1, 0, 2] : (tensor<2x3x2xi2>) -> tensor<3x2x2xi2>
  return %1 : tensor<3x2x2xi2>
}

// i1 is bit packed in DenseElementsAttr and is not folded.
// CHECK-LABEL: func.func @transpose_dense_i1_not_folded
func.func @transpose_dense_i1_not_folded() -> tensor<3x2xi1> {
  // CHECK: stablehlo.transpose
  %0 = stablehlo.constant dense<[[true, false, true], [false, true, false]]> : tensor<2x3xi1>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xi1>) -> tensor<3x2xi1>
  return %1 : tensor<3x2xi1>
}

// A resource whose only user is the transpose is transposed in place and
// keeps its key. Packed i4, two elements per byte with the first element in
// the low nibble: [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]].
// CHECK-LABEL: func.func @transpose_resource_packed_i4
func.func @transpose_resource_packed_i4() -> tensor<3x2xi4> {
  // CHECK: %[[CST:.+]] = stablehlo.constant dense_resource<packed_i4> : tensor<3x2xi4>
  // CHECK-NOT: stablehlo.transpose
  // CHECK: return %[[CST]]
  %0 = stablehlo.constant dense_resource<packed_i4> : tensor<2x3xi4>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xi4>) -> tensor<3x2xi4>
  return %1 : tensor<3x2xi4>
}

// Unpacked i4 blob, one element per byte.
// CHECK-LABEL: func.func @transpose_resource_unpacked_i4
func.func @transpose_resource_unpacked_i4() -> tensor<3x2xi4> {
  // CHECK: stablehlo.constant dense_resource<unpacked_i4> : tensor<3x2xi4>
  // CHECK-NOT: stablehlo.transpose
  %0 = stablehlo.constant dense_resource<unpacked_i4> : tensor<2x3xi4>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xi4>) -> tensor<3x2xi4>
  return %1 : tensor<3x2xi4>
}

// Packed i2 blob whose element count is not a multiple of four:
// [[0, 1, 2], [3, 0, 1], [2, 3, 0]] -> [[0, 3, 2], [1, 0, 3], [2, 1, 0]].
// CHECK-LABEL: func.func @transpose_resource_packed_i2
func.func @transpose_resource_packed_i2() -> tensor<3x3xi2> {
  // CHECK: stablehlo.constant dense_resource<packed_i2> : tensor<3x3xi2>
  // CHECK-NOT: stablehlo.transpose
  %0 = stablehlo.constant dense_resource<packed_i2> : tensor<3x3xi2>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<3x3xi2>) -> tensor<3x3xi2>
  return %1 : tensor<3x3xi2>
}

// A sub-byte blob of an unexpected size is left alone.
// CHECK-LABEL: func.func @transpose_resource_bad_size_i4_not_folded
func.func @transpose_resource_bad_size_i4_not_folded() -> tensor<3x2xi4> {
  // CHECK: stablehlo.transpose
  %0 = stablehlo.constant dense_resource<bad_size_i4> : tensor<2x3xi4>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xi4>) -> tensor<3x2xi4>
  return %1 : tensor<3x2xi4>
}

// Rank 3 i8: 1..12 -> 1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12.
// CHECK-LABEL: func.func @transpose_resource_i8_rank3
func.func @transpose_resource_i8_rank3() -> tensor<3x2x2xi8> {
  // CHECK: stablehlo.constant dense_resource<i8_1_to_12> : tensor<3x2x2xi8>
  // CHECK-NOT: stablehlo.transpose
  %0 = stablehlo.constant dense_resource<i8_1_to_12> : tensor<2x3x2xi8>
  %1 = stablehlo.transpose %0, dims = [1, 0, 2] : (tensor<2x3x2xi8>) -> tensor<3x2x2xi8>
  return %1 : tensor<3x2x2xi8>
}

// The pattern of a head-major einsum weight: transpose, then reshape. Both
// fold into the original blob without copies.
// CHECK-LABEL: func.func @transpose_then_reshape_resource
func.func @transpose_then_reshape_resource() -> tensor<3x4xi4> {
  // CHECK: %[[CST:.+]] = stablehlo.constant dense_resource<packed_i4_1_to_12> : tensor<3x4xi4>
  // CHECK-NOT: stablehlo.transpose
  // CHECK-NOT: stablehlo.reshape
  // CHECK: return %[[CST]]
  %0 = stablehlo.constant dense_resource<packed_i4_1_to_12> : tensor<2x3x2xi4>
  %1 = stablehlo.transpose %0, dims = [1, 0, 2] : (tensor<2x3x2xi4>) -> tensor<3x2x2xi4>
  %2 = stablehlo.reshape %1 : (tensor<3x2x2xi4>) -> tensor<3x4xi4>
  return %2 : tensor<3x4xi4>
}

// A reshape before the transpose reuses the blob too, and the transpose then
// folds in place.
// CHECK-LABEL: func.func @reshape_then_transpose_resource
func.func @reshape_then_transpose_resource() -> tensor<3x2xi8> {
  // CHECK: %[[CST:.+]] = stablehlo.constant dense_resource<i8_six> : tensor<3x2xi8>
  // CHECK-NOT: stablehlo.reshape
  // CHECK-NOT: stablehlo.transpose
  // CHECK: return %[[CST]]
  %0 = stablehlo.constant dense_resource<i8_six> : tensor<6xi8>
  %1 = stablehlo.reshape %0 : (tensor<6xi8>) -> tensor<2x3xi8>
  %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x3xi8>) -> tensor<3x2xi8>
  return %2 : tensor<3x2xi8>
}

// A blob shared by constants in two functions, transposed the same way in
// both, is transposed once in place.
// CHECK-LABEL: func.func @shared_same_perm_0
func.func @shared_same_perm_0() -> tensor<3x2xi8> {
  // CHECK: stablehlo.constant dense_resource<i8_shared_same> : tensor<3x2xi8>
  // CHECK-NOT: stablehlo.transpose
  %0 = stablehlo.constant dense_resource<i8_shared_same> : tensor<2x3xi8>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xi8>) -> tensor<3x2xi8>
  return %1 : tensor<3x2xi8>
}
// CHECK-LABEL: func.func @shared_same_perm_1
func.func @shared_same_perm_1() -> tensor<3x2xi8> {
  // CHECK: stablehlo.constant dense_resource<i8_shared_same> : tensor<3x2xi8>
  // CHECK-NOT: stablehlo.transpose
  %0 = stablehlo.constant dense_resource<i8_shared_same> : tensor<2x3xi8>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xi8>) -> tensor<3x2xi8>
  return %1 : tensor<3x2xi8>
}

// A blob transposed in two different ways gets one copy per result, and the
// source blob is released.
// CHECK-LABEL: func.func @shared_different_perms
func.func @shared_different_perms() -> (tensor<3x2x2xi8>, tensor<2x2x3xi8>) {
  // CHECK-DAG: stablehlo.constant dense_resource<[[$KEY_P102:i8_shared_diff_transpose_1_0_2[_0-9]*]]> : tensor<3x2x2xi8>
  // CHECK-DAG: stablehlo.constant dense_resource<[[$KEY_P021:i8_shared_diff_transpose_0_2_1[_0-9]*]]> : tensor<2x2x3xi8>
  // CHECK-NOT: stablehlo.transpose
  %0 = stablehlo.constant dense_resource<i8_shared_diff> : tensor<2x3x2xi8>
  %1 = stablehlo.transpose %0, dims = [1, 0, 2] : (tensor<2x3x2xi8>) -> tensor<3x2x2xi8>
  %2 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<2x3x2xi8>) -> tensor<2x2x3xi8>
  return %1, %2 : tensor<3x2x2xi8>, tensor<2x2x3xi8>
}

// A blob with a user other than the transpose keeps its data, and the
// transpose gets a copy.
// CHECK-LABEL: func.func @shared_with_other_user
func.func @shared_with_other_user() -> (tensor<2x3xi8>, tensor<3x2xi8>) {
  // CHECK-DAG: %[[ORIG:.+]] = stablehlo.constant dense_resource<i8_other_user> : tensor<2x3xi8>
  // CHECK-DAG: %[[COPY:.+]] = stablehlo.constant dense_resource<[[$KEY_OTHER:i8_other_user_transpose_1_0[_0-9]*]]> : tensor<3x2xi8>
  // CHECK-NOT: stablehlo.transpose
  // CHECK: return %[[ORIG]], %[[COPY]]
  %0 = stablehlo.constant dense_resource<i8_other_user> : tensor<2x3xi8>
  %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xi8>) -> tensor<3x2xi8>
  return %0, %1 : tensor<2x3xi8>, tensor<3x2xi8>
}

// CHECK-LABEL: dialect_resources
// CHECK-DAG: packed_i4: "0x04000000415263"
// CHECK-DAG: unpacked_i4: "0x04000000010402050306"
// CHECK-DAG: packed_i2: "0x040000006C6C00"
// CHECK-DAG: bad_size_i4: "0x0400000021436500"
// CHECK-DAG: i8_1_to_12: "0x04000000010207080304090A05060B0C"
// CHECK-DAG: packed_i4_1_to_12: "0x04000000218743A965CB"
// CHECK-DAG: i8_six: "0x04000000010402050306"
// CHECK-DAG: i8_shared_same: "0x04000000010402050306"
// CHECK-DAG: [[$KEY_P102]]: "0x40000000010207080304090A05060B0C"
// CHECK-DAG: [[$KEY_P021]]: "0x4000000001030502040607090B080A0C"
// CHECK-DAG: i8_other_user: "0x04000000010203040506"
// CHECK-DAG: [[$KEY_OTHER]]: "0x40000000010402050306"
// CHECK-NOT: i8_shared_diff:

{-#
  dialect_resources: {
    builtin: {
      packed_i4: "0x04000000214365",
      unpacked_i4: "0x04000000010203040506",
      packed_i2: "0x04000000E4E400",
      bad_size_i4: "0x0400000021436500",
      i8_1_to_12: "0x040000000102030405060708090A0B0C",
      packed_i4_1_to_12: "0x0400000021436587A9CB",
      i8_six: "0x04000000010203040506",
      i8_shared_same: "0x04000000010203040506",
      i8_shared_diff: "0x040000000102030405060708090A0B0C",
      i8_other_user: "0x04000000010203040506"
    }
  }
#-}
