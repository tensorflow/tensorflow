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

// RUN: fusion_compiler_opt %s -xtile-cpu-legalize-narrow-float-storage \
// RUN:   -split-input-file | FileCheck %s

func.func @extract_extf(%arg0: memref<16xbf16>, %arg1: index) -> tensor<8xf32> {
  %0 = xtile.extract %arg0[%arg1] [8] [1] : memref<16xbf16> -> tensor<8xbf16>
  %1 = arith.extf %0 : tensor<8xbf16> to tensor<8xf32>
  return %1 : tensor<8xf32>
}
// CHECK-LABEL: func.func @extract_extf(
// CHECK-SAME:      %[[ARG0:.*]]: memref<16xbf16>, %[[ARG1:.*]]: index)
// CHECK:         %[[VIEW:.*]] = xtile.memref_bitcast %[[ARG0]] : memref<16xbf16> -> memref<16xi16>
// CHECK:         %[[TILE:.*]] = xtile.extract %[[VIEW]][%[[ARG1]]] [8] [1] : memref<16xi16> -> tensor<8xi16>
// CHECK:         %[[CAST:.*]] = tensor.bitcast %[[TILE]] : tensor<8xi16> to tensor<8xbf16>
// CHECK:         arith.extf %[[CAST]] : tensor<8xbf16> to tensor<8xf32>

// -----

func.func @copy(%arg0: memref<16xbf16>, %arg1: memref<16xbf16>, %arg2: index) {
  %0 = xtile.extract %arg0[%arg2] [8] [1] : memref<16xbf16> -> tensor<8xbf16>
  xtile.insert %0 into %arg1[%arg2] [8] [1] : tensor<8xbf16> -> memref<16xbf16>
  return
}
// CHECK-LABEL: func.func @copy(
// CHECK-SAME:      %[[ARG0:.*]]: memref<16xbf16>, %[[ARG1:.*]]: memref<16xbf16>, %[[ARG2:.*]]: index)
// CHECK-DAG:     %[[SRC:.*]] = xtile.memref_bitcast %[[ARG0]] : memref<16xbf16> -> memref<16xi16>
// CHECK-DAG:     %[[DST:.*]] = xtile.memref_bitcast %[[ARG1]] : memref<16xbf16> -> memref<16xi16>
// CHECK:         %[[TILE:.*]] = xtile.extract %[[SRC]][%[[ARG2]]] [8] [1] : memref<16xi16> -> tensor<8xi16>
// CHECK-NOT:     tensor.bitcast
// CHECK:         xtile.insert %[[TILE]] into %[[DST]][%[[ARG2]]] [8] [1] : tensor<8xi16> -> memref<16xi16>

// -----

// All tile accesses to a buffer share one view of it.
func.func @shared_view(%arg0: memref<16xf16>, %arg1: index, %arg2: index) -> tensor<8xf32> {
  %0 = xtile.extract %arg0[%arg1] [8] [1] : memref<16xf16> -> tensor<8xf16>
  %1 = xtile.extract %arg0[%arg2] [8] [1] : memref<16xf16> -> tensor<8xf16>
  %2 = arith.extf %0 : tensor<8xf16> to tensor<8xf32>
  %3 = arith.extf %1 : tensor<8xf16> to tensor<8xf32>
  %4 = arith.addf %2, %3 : tensor<8xf32>
  return %4 : tensor<8xf32>
}
// CHECK-LABEL: func.func @shared_view(
// CHECK:         %[[VIEW:.*]] = xtile.memref_bitcast %{{.*}} : memref<16xf16> -> memref<16xi16>
// CHECK-NOT:     xtile.memref_bitcast
// CHECK:         xtile.extract %[[VIEW]]
// CHECK:         xtile.extract %[[VIEW]]

// -----

// The transpose of the truncated tile runs on the storage type.
func.func @truncf_transpose(%arg0: memref<8x8xf32>, %arg1: memref<8x8xbf16>, %arg2: index) {
  %0 = xtile.extract %arg0[%arg2, %arg2] [8, 8] [1, 1] : memref<8x8xf32> -> tensor<8x8xf32>
  %1 = arith.truncf %0 : tensor<8x8xf32> to tensor<8x8xbf16>
  %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<8x8xbf16>) -> tensor<8x8xbf16>
  xtile.insert %2 into %arg1[%arg2, %arg2] [8, 8] [1, 1] : tensor<8x8xbf16> -> memref<8x8xbf16>
  return
}
// CHECK-LABEL: func.func @truncf_transpose(
// CHECK-SAME:      %[[ARG0:.*]]: memref<8x8xf32>, %[[ARG1:.*]]: memref<8x8xbf16>, %[[ARG2:.*]]: index)
// CHECK:         %[[DST:.*]] = xtile.memref_bitcast %[[ARG1]] : memref<8x8xbf16> -> memref<8x8xi16>
// CHECK:         %[[TILE:.*]] = xtile.extract %[[ARG0]]
// CHECK:         %[[TRUNC:.*]] = arith.truncf %[[TILE]] : tensor<8x8xf32> to tensor<8x8xbf16>
// CHECK:         %[[CAST:.*]] = tensor.bitcast %[[TRUNC]] : tensor<8x8xbf16> to tensor<8x8xi16>
// CHECK:         %[[TRANS:.*]] = stablehlo.transpose %[[CAST]], dims = [1, 0] : (tensor<8x8xi16>) -> tensor<8x8xi16>
// CHECK:         xtile.insert %[[TRANS]] into %[[DST]]

// -----

// A chain of data movement ops between tile accesses runs entirely on the
// storage type, without any bitcasts in between.
func.func @data_movement_chain(%arg0: memref<4x8xf8E4M3FN>, %arg1: memref<8x8xf8E4M3FN>, %arg2: index) {
  %0 = xtile.extract %arg0[%arg2, %arg2] [4, 8] [1, 1] : memref<4x8xf8E4M3FN> -> tensor<4x8xf8E4M3FN>
  %1 = stablehlo.concatenate %0, %0, dim = 0 : (tensor<4x8xf8E4M3FN>, tensor<4x8xf8E4M3FN>) -> tensor<8x8xf8E4M3FN>
  %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<8x8xf8E4M3FN>) -> tensor<8x8xf8E4M3FN>
  %3 = stablehlo.reshape %2 : (tensor<8x8xf8E4M3FN>) -> tensor<64xf8E4M3FN>
  %4 = stablehlo.slice %3 [0:32] : (tensor<64xf8E4M3FN>) -> tensor<32xf8E4M3FN>
  %5 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<32xf8E4M3FN>) -> tensor<2x32xf8E4M3FN>
  %6 = stablehlo.reshape %5 : (tensor<2x32xf8E4M3FN>) -> tensor<8x8xf8E4M3FN>
  xtile.insert %6 into %arg1[%arg2, %arg2] [8, 8] [1, 1] : tensor<8x8xf8E4M3FN> -> memref<8x8xf8E4M3FN>
  return
}
// CHECK-LABEL: func.func @data_movement_chain(
// CHECK-NOT:     tensor.bitcast
// CHECK:         %[[TILE:.*]] = xtile.extract {{.*}} : memref<4x8xi8> -> tensor<4x8xi8>
// CHECK:         %[[CONCAT:.*]] = stablehlo.concatenate %[[TILE]], %[[TILE]], dim = 0 : (tensor<4x8xi8>, tensor<4x8xi8>) -> tensor<8x8xi8>
// CHECK:         %[[TRANS:.*]] = stablehlo.transpose %[[CONCAT]], dims = [1, 0] : (tensor<8x8xi8>) -> tensor<8x8xi8>
// CHECK:         %[[RESHAPE:.*]] = stablehlo.reshape %[[TRANS]] : (tensor<8x8xi8>) -> tensor<64xi8>
// CHECK:         %[[SLICE:.*]] = stablehlo.slice %[[RESHAPE]] [0:32] : (tensor<64xi8>) -> tensor<32xi8>
// CHECK:         %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[SLICE]], dims = [1] : (tensor<32xi8>) -> tensor<2x32xi8>
// CHECK:         %[[RESHAPE2:.*]] = stablehlo.reshape %[[BCAST]] : (tensor<2x32xi8>) -> tensor<8x8xi8>
// CHECK:         xtile.insert %[[RESHAPE2]] into {{.*}} : tensor<8x8xi8> -> memref<8x8xi8>
// CHECK-NOT:     tensor.bitcast

// -----

func.func @select(%arg0: memref<16xbf16>, %arg1: memref<16xbf16>, %arg2: tensor<8xi1>, %arg3: index) {
  %0 = xtile.extract %arg0[%arg3] [8] [1] : memref<16xbf16> -> tensor<8xbf16>
  %1 = xtile.extract %arg1[%arg3] [8] [1] : memref<16xbf16> -> tensor<8xbf16>
  %2 = arith.select %arg2, %0, %1 : tensor<8xi1>, tensor<8xbf16>
  xtile.insert %2 into %arg0[%arg3] [8] [1] : tensor<8xbf16> -> memref<16xbf16>
  return
}
// CHECK-LABEL: func.func @select(
// CHECK-SAME:      %[[ARG0:.*]]: memref<16xbf16>, %[[ARG1:.*]]: memref<16xbf16>, %[[COND:.*]]: tensor<8xi1>, %[[ARG3:.*]]: index)
// CHECK-DAG:     %[[VIEW0:.*]] = xtile.memref_bitcast %[[ARG0]]
// CHECK-DAG:     %[[VIEW1:.*]] = xtile.memref_bitcast %[[ARG1]]
// CHECK:         %[[LHS:.*]] = xtile.extract %[[VIEW0]]
// CHECK:         %[[RHS:.*]] = xtile.extract %[[VIEW1]]
// CHECK:         %[[SEL:.*]] = arith.select %[[COND]], %[[LHS]], %[[RHS]] : tensor<8xi1>, tensor<8xi16>
// CHECK:         xtile.insert %[[SEL]] into %[[VIEW0]]

// -----

func.func @mask(%arg0: memref<16xbf16>, %arg1: memref<16xbf16>, %arg2: index) {
  %cst = arith.constant 0.0 : bf16
  %0 = xtile.extract %arg0[%arg2] [8] [1] : memref<16xbf16> -> tensor<8xbf16>
  %1 = xtile.mask %0 bounds [5], %cst : tensor<8xbf16>
  xtile.insert %1 into %arg1[%arg2] [8] [1] : tensor<8xbf16> -> memref<16xbf16>
  return
}
// CHECK-LABEL: func.func @mask(
// CHECK:         %[[ZERO:.*]] = arith.constant 0 : i16
// CHECK:         %[[TILE:.*]] = xtile.extract {{.*}} : memref<16xi16> -> tensor<8xi16>
// CHECK:         %[[MASKED:.*]] = xtile.mask %[[TILE]] bounds [5], %[[ZERO]] : tensor<8xi16>
// CHECK:         xtile.insert %[[MASKED]] into {{.*}} : tensor<8xi16> -> memref<16xi16>

// -----

// Types that are not 8- or 16-bit floats are left alone.
func.func @not_narrow(%arg0: memref<16xf32>, %arg1: memref<16xf4E2M1FN>, %arg2: index) -> (tensor<8xf32>, tensor<8xf4E2M1FN>) {
  %0 = xtile.extract %arg0[%arg2] [8] [1] : memref<16xf32> -> tensor<8xf32>
  %1 = xtile.extract %arg1[%arg2] [8] [1] : memref<16xf4E2M1FN> -> tensor<8xf4E2M1FN>
  return %0, %1 : tensor<8xf32>, tensor<8xf4E2M1FN>
}
// CHECK-LABEL: func.func @not_narrow(
// CHECK-NOT:     xtile.memref_bitcast
// CHECK:         xtile.extract %{{.*}} : memref<16xf32> -> tensor<8xf32>
// CHECK:         xtile.extract %{{.*}} : memref<16xf4E2M1FN> -> tensor<8xf4E2M1FN>
