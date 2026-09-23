// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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

// RUN: fusion_compiler_opt %s -xtile-cpu-lower-memref-bitcast \
// RUN:   -split-input-file | FileCheck %s

func.func @memref_bitcast(%arg0: memref<16xbf16>, %arg1: index) -> vector<8xi16> {
  %c0 = arith.constant 0 : i16
  %0 = xtile.memref_bitcast %arg0 : memref<16xbf16> -> memref<16xi16>
  %1 = vector.transfer_read %0[%arg1], %c0 : memref<16xi16>, vector<8xi16>
  return %1 : vector<8xi16>
}
// CHECK-LABEL: func.func @memref_bitcast(
// CHECK-SAME:      %[[ARG0:.*]]: memref<16xbf16>, %[[ARG1:.*]]: index)
// CHECK-NOT:     xtile.memref_bitcast
// CHECK:         %[[DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<16xbf16> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:         %[[VIEW:.*]] = builtin.unrealized_conversion_cast %[[DESC]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<16xi16>
// CHECK:         vector.transfer_read %[[VIEW]][%[[ARG1]]]

// -----

func.func @memref_bitcast_2d(%arg0: memref<4x8xf8E4M3FN>) -> memref<4x8xi8> {
  %0 = xtile.memref_bitcast %arg0 : memref<4x8xf8E4M3FN> -> memref<4x8xi8>
  return %0 : memref<4x8xi8>
}
// CHECK-LABEL: func.func @memref_bitcast_2d(
// CHECK-SAME:      %[[ARG0:.*]]: memref<4x8xf8E4M3FN>)
// CHECK-NOT:     xtile.memref_bitcast
// CHECK:         %[[DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<4x8xf8E4M3FN> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:         %[[VIEW:.*]] = builtin.unrealized_conversion_cast %[[DESC]] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<4x8xi8>
// CHECK:         return %[[VIEW]]
