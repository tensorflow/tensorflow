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
// RUN: emitters_opt %s -one-shot-bufferize -canonicalize -cse \
// RUN: -split-input-file | FileCheck %s

// CHECK: @extract_strided(%[[SOURCE:.*]]: memref<16xf32>, %[[OFFSET:.*]]: index)
func.func @extract_strided(%source: memref<16xf32>, %tile_id: index) -> tensor<8xf32> {
  // CHECK-DAG: %[[PAD:.*]] = arith.constant dense<0x7FC00000> : vector<8xf32>
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[C15:.*]] = arith.constant 15 : index

  // CHECK: %[[SHIFT:.*]] = arith.subi %[[C15]], %[[OFFSET]] : index
  // CHECK: %[[STRIDED_SHIFT:.*]] = arith.divsi %[[SHIFT]], %[[C2]] : index
  // CHECK: %[[ELEMENTS_TO_END:.*]] = arith.addi %[[STRIDED_SHIFT]], %[[C1]] : index
  // CHECK: %[[SIZE:.*]] = arith.minsi %[[ELEMENTS_TO_END]], %[[C8]] : index
  // CHECK: %[[IS_FULL_TILE:.*]] = arith.cmpi eq, %[[SIZE]], %[[C8]] : index

  // CHECK: %[[BUFFER:.*]] = scf.if %[[IS_FULL_TILE]] -> (memref<8xf32>) {
    // CHECK: %[[STATIC_SUBVIEW:.*]] = memref.subview %[[SOURCE]][%[[OFFSET]]] [8] [2]
    // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<8xf32>
    // CHECK: memref.copy %[[STATIC_SUBVIEW]], %[[ALLOC_0]]
    // CHECK: scf.yield %[[ALLOC_0]] : memref<8xf32>
  // CHECK: } else {
    // CHECK: %[[INPUT_SUBVIEW:.*]] = memref.subview %[[SOURCE]]
    // CHECK-SAME: [%[[OFFSET]]] [%[[SIZE]]] [2]
    // CHECK-SAME: : memref<16xf32> to memref<?xf32, strided<[2], offset: ?>>

    // CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<8xf32>
    // CHECK: vector.store %[[PAD]], %[[ALLOC_1]][%{{.*}}] : memref<8xf32>, vector<8xf32>

    // CHECK: %[[ALLOC_1_SUBVIEW:.*]] = memref.subview %[[ALLOC_1]]
    // CHECK-SAME: [0] [%[[SIZE]]] [1] : memref<8xf32> to memref<?xf32, strided<[1]>>

    // CHECK: memref.copy %[[INPUT_SUBVIEW]], %[[ALLOC_1_SUBVIEW]]
    // CHECK-SAME: : memref<?xf32, strided<[2], offset: ?>> to memref<?xf32, strided<[1]>>
    // CHECK: scf.yield %[[ALLOC_1]] : memref<8xf32>
  // CHECK: }

  // CHECK: %[[TILE:.*]] = bufferization.to_tensor %[[BUFFER]]
  // CHECK-SAME: : memref<8xf32> to tensor<8xf32>
  %tile = xtile.extract %source[%tile_id][8][2] : memref<16xf32> -> tensor<8xf32>
  // CHECK: return %[[TILE]] : tensor<8xf32>
  return %tile : tensor<8xf32>
}

// -----

// CHECK: @insert_strided(
// CHECK-SAME: %[[SOURCE:.*]]: tensor<8xf32>,
// CHECK-SAME: %[[DESTINATION:.*]]: memref<16xf32>,
// CHECK-SAME: %[[OFFSET:.*]]: index)
func.func @insert_strided(%source: tensor<8xf32>, %destination: memref<16xf32>, %tile_id: index) {
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[C15:.*]] = arith.constant 15 : index

  // CHECK: %[[SOURCE_BUFFER:.*]] = bufferization.to_buffer %[[SOURCE]]
  // CHECK-SAME: : tensor<8xf32> to memref<8xf32, strided<[?], offset: ?>>

  // CHECK: %[[SHIFT:.*]] = arith.subi %[[C15]], %[[OFFSET]] : index
  // CHECK: %[[STRIDED_SHIFT:.*]] = arith.divsi %[[SHIFT]], %[[C2]] : index
  // CHECK: %[[ELEMENTS_TO_END:.*]] = arith.addi %[[STRIDED_SHIFT]], %[[C1]] : index
  // CHECK: %[[SIZE:.*]] = arith.minsi %[[ELEMENTS_TO_END]], %[[C8]] : index
  // CHECK: %[[IS_FULL_TILE:.*]] = arith.cmpi eq, %[[SIZE]], %[[C8]] : index

  // CHECK: scf.if %[[IS_FULL_TILE]] {
    // CHECK:   %[[DESTINATION_SUBVIEW:.*]] = memref.subview %[[DESTINATION]][%[[OFFSET]]] [8] [2]
    // CHECK:   memref.copy %[[SOURCE_BUFFER]], %[[DESTINATION_SUBVIEW]]
  // CHECK: } else {
    // CHECK: %[[SOURCE_SUBVIEW:.*]] = memref.subview %[[SOURCE_BUFFER]][0] [%[[SIZE]]] [1]
    // CHECK-SAME: : memref<8xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>

    // CHECK: %[[DESTINATION_SUBVIEW:.*]] = memref.subview %[[DESTINATION]]
    // CHECK-SAME: [%[[OFFSET]]] [%[[SIZE]]] [2]
    // CHECK-SAME: : memref<16xf32> to memref<?xf32, strided<[2], offset: ?>>

    // CHECK: memref.copy %[[SOURCE_SUBVIEW]], %[[DESTINATION_SUBVIEW]]
    // CHECK-SAME: : memref<?xf32, strided<[?], offset: ?>>
    // CHECK-SAME: to memref<?xf32, strided<[2], offset: ?>>
  // CHECK: }

  xtile.insert %source into %destination[%tile_id][8][2] : tensor<8xf32> -> memref<16xf32>
  return
}

// -----

// CHECK: @extract_identity(%[[SOURCE:.*]]: memref<16xf32>, %[[OFFSET:.*]]: index)
func.func @extract_identity(%source: memref<16xf32>, %tile_id: index) -> tensor<8xf32> {
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[DIFF:.*]] = arith.subi %[[C16]], %[[OFFSET]] : index
  // CHECK-DAG: %[[SIZE:.*]] = arith.minsi %[[DIFF]], %[[C8]] : index
  // CHECK-DAG: %[[IS_FULL_TILE:.*]] = arith.cmpi eq, %[[SIZE]], %[[C8]] : index

  // CHECK: %[[BUFFER:.*]] = scf.if %[[IS_FULL_TILE]] -> (memref<8xf32>) {
  // CHECK:   %[[SUBVIEW:.*]] = memref.subview %[[SOURCE]][%[[OFFSET]]] [8] [1] : memref<16xf32> to memref<8xf32, strided<[1], offset: ?>>
  // CHECK:   %[[ALLOC_IF:.*]] = memref.alloc() : memref<8xf32>
  // CHECK:   memref.copy %[[SUBVIEW]], %[[ALLOC_IF]] : memref<8xf32, strided<[1], offset: ?>> to memref<8xf32>
  // CHECK:   scf.yield %[[ALLOC_IF]] : memref<8xf32>
  // CHECK: } else {
  // CHECK:   %[[INPUT_SUBVIEW:.*]] = memref.subview %[[SOURCE]][%[[OFFSET]]] [%[[SIZE]]] [1] : memref<16xf32> to memref<?xf32, strided<[1], offset: ?>>
  // CHECK:   %[[ALLOC_ELSE:.*]] = memref.alloc() : memref<8xf32>
  // CHECK:   vector.store %{{.*}}, %[[ALLOC_ELSE]][%{{.*}}] : memref<8xf32>, vector<8xf32>
  // CHECK:   %[[ALLOC_SUBVIEW:.*]] = memref.subview %[[ALLOC_ELSE]][0] [%[[SIZE]]] [1] : memref<8xf32> to memref<?xf32, strided<[1]>>
  // CHECK:   memref.copy %[[INPUT_SUBVIEW]], %[[ALLOC_SUBVIEW]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
  // CHECK:   scf.yield %[[ALLOC_ELSE]] : memref<8xf32>
  // CHECK: }
  // CHECK: %[[TILE:.*]] = bufferization.to_tensor %[[BUFFER]] : memref<8xf32> to tensor<8xf32>
  %tile = xtile.extract %source[%tile_id][8][1] : memref<16xf32> -> tensor<8xf32>
  return %tile : tensor<8xf32>
}

// -----

// CHECK: @extract_5d(%[[SOURCE:.*]]: memref<1x2x1x32768x256xf32>, %[[D1:.*]]: index, %[[D3:.*]]: index, %[[D4:.*]]: index)
func.func @extract_5d(%source: memref<1x2x1x32768x256xf32>, %d1: index, %d3: index, %d4: index) -> tensor<1x1x1x1x16xf32> {
  // CHECK-DAG: %[[C_0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C_16:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[C_256:.*]] = arith.constant 256 : index
  // CHECK: %[[DIFF_4:.*]] = arith.subi %[[C_256]], %[[D4]] : index
  // CHECK: %[[SIZE_4:.*]] = arith.minsi %[[DIFF_4]], %[[C_16]] : index
  // CHECK: %[[IS_FULL_TILE:.*]] = arith.cmpi eq, %[[SIZE_4]], %[[C_16]] : index
  // CHECK: %[[BUFFER:.*]] = scf.if %[[IS_FULL_TILE]] -> (memref<1x1x1x1x16xf32>) {
  // CHECK:   %[[SUBVIEW:.*]] = memref.subview %[[SOURCE]][%[[C_0]], %[[D1]], %[[C_0]], %[[D3]], %[[D4]]] [1, 1, 1, 1, 16] [0, 1, 0, 1, 1] : memref<1x2x1x32768x256xf32> to memref<1x1x1x1x16xf32, strided<[0, 8388608, 0, 256, 1], offset: ?>>
  // CHECK:   %[[ALLOC_IF:.*]] = memref.alloc() : memref<1x1x1x1x16xf32>
  // CHECK:   memref.copy %[[SUBVIEW]], %[[ALLOC_IF]]
  // CHECK:   scf.yield %[[ALLOC_IF]] : memref<1x1x1x1x16xf32>
  // CHECK: } else {
  // CHECK:   %[[INPUT_SUBVIEW:.*]] = memref.subview %[[SOURCE]][%[[C_0]], %[[D1]], %[[C_0]], %[[D3]], %[[D4]]] [1, 1, 1, 1, %[[SIZE_4]]] [0, 1, 0, 1, 1] : memref<1x2x1x32768x256xf32> to memref<1x1x1x1x?xf32, strided<[0, 8388608, 0, 256, 1], offset: ?>>
  // CHECK:   %[[ALLOC_ELSE:.*]] = memref.alloc() : memref<1x1x1x1x16xf32>
  // CHECK:   vector.store %{{.*}}, %[[ALLOC_ELSE]][%[[C_0]], %[[C_0]], %[[C_0]], %[[C_0]], %[[C_0]]] : memref<1x1x1x1x16xf32>, vector<16xf32>
  // CHECK:   %[[ALLOC_SUBVIEW:.*]] = memref.subview %[[ALLOC_ELSE]][0, 0, 0, 0, 0] [1, 1, 1, 1, %[[SIZE_4]]] [1, 1, 1, 1, 1] : memref<1x1x1x1x16xf32> to memref<1x1x1x1x?xf32, strided<[16, 16, 16, 16, 1]>>
  // CHECK:   memref.copy %[[INPUT_SUBVIEW]], %[[ALLOC_SUBVIEW]]
  // CHECK:   scf.yield %[[ALLOC_ELSE]] : memref<1x1x1x1x16xf32>
  // CHECK: }

  // CHECK: bufferization.to_tensor %[[BUFFER]] : memref<1x1x1x1x16xf32> to tensor<1x1x1x1x16xf32>
  %c0 = arith.constant 0 : index
  %tile = xtile.extract %source[%c0, %d1, %c0, %d3, %d4][1, 1, 1, 1, 16][0, 1, 0, 1, 1] : memref<1x2x1x32768x256xf32> -> tensor<1x1x1x1x16xf32>
  return %tile : tensor<1x1x1x1x16xf32>
}

// -----

// CHECK: @extract_static(%[[SOURCE:.*]]: memref<16xf32>)
func.func @extract_static(%source: memref<16xf32>) -> tensor<8xf32> {
  // CHECK-NOT: scf.if
  // CHECK-NOT: vector.store
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %[[SOURCE]][0] [8] [1] : memref<16xf32> to memref<8xf32, strided<[1]>>
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<8xf32>
  // CHECK: memref.copy %[[SUBVIEW]], %[[ALLOC]] : memref<8xf32, strided<[1]>> to memref<8xf32>
  // CHECK: %[[TILE:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<8xf32> to tensor<8xf32>
  // CHECK: return %[[TILE]]
  %c0 = arith.constant 0 : index
  %tile = xtile.extract %source[%c0][8][1] : memref<16xf32> -> tensor<8xf32>
  return %tile : tensor<8xf32>
}

// -----

#map = #xla.indexing_map<"(d0) -> (d0 * 8), domain: d0 in [0, 1]">
// CHECK: #[[MAP:.*]] = #xla.indexing_map<"(d0) -> (d0 * 8), domain: d0 in [0, 1]">
// CHECK: @insert_in_bounds(
// CHECK-SAME: %[[SOURCE:.*]]: tensor<8xf32>,
// CHECK-SAME: %[[DESTINATION:.*]]: memref<16xf32>,
// CHECK-SAME: %[[TILE_ID:.*]]: index)
func.func @insert_in_bounds(%source: tensor<8xf32>, %destination: memref<16xf32>, %tile_id: index) {
  // CHECK-NOT: scf.if
  // CHECK: %[[SOURCE_BUFFER:.*]] = bufferization.to_buffer %[[SOURCE]] : tensor<8xf32> to memref<8xf32, strided<[?], offset: ?>>
  // CHECK: %[[OFFSET:.*]] = xla.apply_indexing #[[MAP]](%[[TILE_ID]])
  // CHECK: %[[DESTINATION_SUBVIEW:.*]] = memref.subview %[[DESTINATION]][%[[OFFSET]]] [8] [1] : memref<16xf32> to memref<8xf32, strided<[1], offset: ?>>
  // CHECK: memref.copy %[[SOURCE_BUFFER]], %[[DESTINATION_SUBVIEW]] : memref<8xf32, strided<[?], offset: ?>> to memref<8xf32, strided<[1], offset: ?>>
  %offset = xla.apply_indexing #map(%tile_id)
  xtile.insert %source into %destination[%offset][8][1] : tensor<8xf32> -> memref<16xf32>
  return
}

// -----

#map = #xla.indexing_map<"(d0) -> (d0 * 2), domain: d0 in [0, 1]">
// CHECK: #[[MAP:.*]] = #xla.indexing_map<"(d0) -> (d0 * 2), domain: d0 in [0, 1]">
// CHECK: @extract_strided_in_bounds(%[[SOURCE:.*]]: memref<17xf32>, %[[TILE_ID:.*]]: index)
func.func @extract_strided_in_bounds(%source: memref<17xf32>, %tile_id: index) -> tensor<8xf32> {
  // CHECK-NOT: scf.if
  // CHECK-NOT: vector.store
  // CHECK: %[[OFFSET:.*]] = xla.apply_indexing #[[MAP]](%[[TILE_ID]])
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %[[SOURCE]][%[[OFFSET]]] [8] [2] : memref<17xf32> to memref<8xf32, strided<[2], offset: ?>>
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<8xf32>
  // CHECK: memref.copy %[[SUBVIEW]], %[[ALLOC]] : memref<8xf32, strided<[2], offset: ?>> to memref<8xf32>
  // CHECK: %[[TILE:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<8xf32> to tensor<8xf32>
  // CHECK: return %[[TILE]] : tensor<8xf32>
  %offset = xla.apply_indexing #map(%tile_id)
  %tile = xtile.extract %source[%offset][8][2] : memref<17xf32> -> tensor<8xf32>
  return %tile : tensor<8xf32>
}

// -----

#map = #xla.indexing_map<"(d0) -> (d0 * 2), domain: d0 in [0, 1]">
// CHECK: #[[MAP:.*]] = #xla.indexing_map<"(d0) -> (d0 * 2), domain: d0 in [0, 1]">
// CHECK: @extract_strided_out_of_bounds(%[[SOURCE:.*]]: memref<16xf32>, %[[TILE_ID:.*]]: index)
func.func @extract_strided_out_of_bounds(%source: memref<16xf32>, %tile_id: index) -> tensor<8xf32> {
  // CHECK: %[[OFFSET:.*]] = xla.apply_indexing #[[MAP]](%[[TILE_ID]])
  // CHECK: scf.if
  %offset = xla.apply_indexing #map(%tile_id)
  %tile = xtile.extract %source[%offset][8][2] : memref<16xf32> -> tensor<8xf32>
  return %tile : tensor<8xf32>
}

// -----

#map = #xla.indexing_map<"(d0) -> (d0 * 4), domain: d0 in [0, 1]">
// CHECK: #[[MAP:.*]] = #xla.indexing_map<"(d0) -> (d0 * 4), domain: d0 in [0, 1]">
// CHECK: @extract_rank_reduced(%[[SOURCE:.*]]: memref<2x8x16xf32>, %[[D0:.*]]: index, %[[D1:.*]]: index, %[[D2:.*]]: index)
func.func @extract_rank_reduced(%source: memref<2x8x16xf32>, %d0: index, %d1: index, %d2: index) -> tensor<4x8xf32> {
  // CHECK-DAG: %[[C_8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[C_16:.*]] = arith.constant 16 : index
  // CHECK: %[[OFFSET_1:.*]] = xla.apply_indexing #[[MAP]](%[[D1]])
  // CHECK: %[[DIFF_2:.*]] = arith.subi %[[C_16]], %[[D2]] : index
  // CHECK: %[[SIZE_2:.*]] = arith.minsi %[[DIFF_2]], %[[C_8]] : index
  // CHECK: %[[IS_FULL_TILE:.*]] = arith.cmpi eq, %[[SIZE_2]], %[[C_8]] : index
  // CHECK: %[[BUFFER:.*]] = scf.if %[[IS_FULL_TILE]] -> (memref<4x8xf32>) {
  // CHECK:   %[[SUBVIEW:.*]] = memref.subview %[[SOURCE]][%[[D0]], %[[OFFSET_1]], %[[D2]]] [1, 4, 8] [1, 1, 1] : memref<2x8x16xf32> to memref<4x8xf32, strided<[16, 1], offset: ?>>
  // CHECK:   %[[ALLOC_IF:.*]] = memref.alloc() : memref<4x8xf32>
  // CHECK:   memref.copy %[[SUBVIEW]], %[[ALLOC_IF]]
  // CHECK:   scf.yield %[[ALLOC_IF]] : memref<4x8xf32>
  // CHECK: } else {
  // CHECK:   %[[INPUT_SUBVIEW:.*]] = memref.subview %[[SOURCE]][%[[D0]], %[[OFFSET_1]], %[[D2]]] [1, 4, %[[SIZE_2]]] [1, 1, 1] : memref<2x8x16xf32> to memref<4x?xf32, strided<[16, 1], offset: ?>>
  // CHECK:   %[[ALLOC_ELSE:.*]] = memref.alloc() : memref<4x8xf32>
  // CHECK:   vector.store %{{.*}}, %[[ALLOC_ELSE]][%{{.*}}, %{{.*}}] : memref<4x8xf32>, vector<32xf32>
  // CHECK:   %[[ALLOC_SUBVIEW:.*]] = memref.subview %[[ALLOC_ELSE]][0, 0] [4, %[[SIZE_2]]] [1, 1] : memref<4x8xf32> to memref<4x?xf32, strided<[8, 1]>>
  // CHECK:   memref.copy %[[INPUT_SUBVIEW]], %[[ALLOC_SUBVIEW]]
  // CHECK:   scf.yield %[[ALLOC_ELSE]] : memref<4x8xf32>
  // CHECK: }
  // CHECK: bufferization.to_tensor %[[BUFFER]] : memref<4x8xf32> to tensor<4x8xf32>
  %offset_1 = xla.apply_indexing #map(%d1)
  %tile = xtile.extract %source[%d0, %offset_1, %d2][1, 4, 8][1, 1, 1] : memref<2x8x16xf32> -> tensor<4x8xf32>
  return %tile : tensor<4x8xf32>
}

// -----

#map = #xla.indexing_map<"(d0) -> (d0 * 4), domain: d0 in [0, 1]">
// CHECK: #[[MAP:.*]] = #xla.indexing_map<"(d0) -> (d0 * 4), domain: d0 in [0, 1]">
// CHECK: @insert_rank_reduced(
// CHECK-SAME: %[[SOURCE:.*]]: tensor<4x1x8xf32>,
// CHECK-SAME: %[[DESTINATION:.*]]: memref<2x8x2x16xf32>,
// CHECK-SAME: %[[D0:.*]]: index, %[[D1:.*]]: index, %[[D2:.*]]: index, %[[D3:.*]]: index)
func.func @insert_rank_reduced(%source: tensor<4x1x8xf32>, %destination: memref<2x8x2x16xf32>, %d0: index, %d1: index, %d2: index, %d3: index) {
  // CHECK-DAG: %[[C_8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[C_16:.*]] = arith.constant 16 : index
  // CHECK: %[[SOURCE_BUFFER:.*]] = bufferization.to_buffer %[[SOURCE]] : tensor<4x1x8xf32> to memref<4x1x8xf32, strided<[?, ?, ?], offset: ?>>
  // CHECK: %[[OFFSET_1:.*]] = xla.apply_indexing #[[MAP]](%[[D1]])
  // CHECK: %[[DIFF_3:.*]] = arith.subi %[[C_16]], %[[D3]] : index
  // CHECK: %[[SIZE_3:.*]] = arith.minsi %[[DIFF_3]], %[[C_8]] : index
  // CHECK: %[[IS_FULL_TILE:.*]] = arith.cmpi eq, %[[SIZE_3]], %[[C_8]] : index
  // CHECK: scf.if %[[IS_FULL_TILE]] {
  // CHECK:   %[[DESTINATION_SUBVIEW:.*]] = memref.subview %[[DESTINATION]][%[[D0]], %[[OFFSET_1]], %[[D2]], %[[D3]]] [1, 4, 1, 8] [1, 1, 1, 1] : memref<2x8x2x16xf32> to memref<4x1x8xf32, strided<[32, 16, 1], offset: ?>>
  // CHECK:   memref.copy %[[SOURCE_BUFFER]], %[[DESTINATION_SUBVIEW]]
  // CHECK: } else {
  // CHECK:   %[[SOURCE_SUBVIEW:.*]] = memref.subview %[[SOURCE_BUFFER]][0, 0, 0] [4, 1, %[[SIZE_3]]] [1, 1, 1] : memref<4x1x8xf32, strided<[?, ?, ?], offset: ?>> to memref<4x1x?xf32, strided<[?, ?, ?], offset: ?>>
  // CHECK:   %[[DESTINATION_SUBVIEW:.*]] = memref.subview %[[DESTINATION]][%[[D0]], %[[OFFSET_1]], %[[D2]], %[[D3]]] [1, 4, 1, %[[SIZE_3]]] [1, 1, 1, 1] : memref<2x8x2x16xf32> to memref<4x1x?xf32, strided<[32, 16, 1], offset: ?>>
  // CHECK:   memref.copy %[[SOURCE_SUBVIEW]], %[[DESTINATION_SUBVIEW]]
  // CHECK: }
  %offset_1 = xla.apply_indexing #map(%d1)
  xtile.insert %source into %destination[%d0, %offset_1, %d2, %d3][1, 4, 1, 8][1, 1, 1, 1] : tensor<4x1x8xf32> -> memref<2x8x2x16xf32>
  return
}

// -----

// CHECK-LABEL: @extract_int(
func.func @extract_int(%source: memref<16xi32>, %tile_id: index) -> tensor<8xi32> {
  // CHECK-DAG: %[[PAD:.*]] = arith.constant dense<-1> : vector<8xi32>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: scf.if {{.*}} -> (memref<8xi32>) {
  // CHECK: } else {
  // CHECK:   %[[ALLOC_ELSE:.*]] = memref.alloc() : memref<8xi32>
  // CHECK:   vector.store %[[PAD]], %[[ALLOC_ELSE]][%[[C0]]] : memref<8xi32>, vector<8xi32>
  // CHECK: }
  %tile = xtile.extract %source[%tile_id][8][1] : memref<16xi32> -> tensor<8xi32>
  return %tile : tensor<8xi32>
}

// -----

// CHECK-LABEL: @extract_complex(
func.func @extract_complex(%source: memref<16xcomplex<f32>>, %tile_id: index) -> tensor<8xcomplex<f32>> {
  // CHECK-DAG: %[[NAN:.*]] = arith.constant 0x7FC00000 : f32
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK: scf.if {{.*}} -> (memref<8xcomplex<f32>>) {
  // CHECK: } else {
  // CHECK:   %[[ALLOC_ELSE:.*]] = memref.alloc() : memref<8xcomplex<f32>>
  // CHECK:   %[[PAD:.*]] = complex.create %[[NAN]], %[[NAN]] : complex<f32>
  // CHECK:   scf.for %[[IV:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
  // CHECK:     memref.store %[[PAD]], %[[ALLOC_ELSE]][%[[IV]]] : memref<8xcomplex<f32>>
  // CHECK:   }
  // CHECK: }
  %tile = xtile.extract %source[%tile_id][8][1] : memref<16xcomplex<f32>> -> tensor<8xcomplex<f32>>
  return %tile : tensor<8xcomplex<f32>>
}

// -----

// CHECK-LABEL: @extract_subbyte_int(
func.func @extract_subbyte_int(%source: memref<4x16xi4>, %d0: index, %d1: index) -> tensor<2x8xi4> {
  // CHECK-DAG: %[[PAD:.*]] = arith.constant -1 : i4
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK: scf.if {{.*}} -> (memref<2x8xi4>) {
  // CHECK: } else {
  // CHECK:   %[[ALLOC_ELSE:.*]] = memref.alloc() : memref<2x8xi4>
  // CHECK:   scf.for %[[IV0:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
  // CHECK:     scf.for %[[IV1:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
  // CHECK:       memref.store %[[PAD]], %[[ALLOC_ELSE]][%[[IV0]], %[[IV1]]] : memref<2x8xi4>
  // CHECK:     }
  // CHECK:   }
  // CHECK: }
  %tile = xtile.extract %source[%d0, %d1][2, 8][1, 1] : memref<4x16xi4> -> tensor<2x8xi4>
  return %tile : tensor<2x8xi4>
}

// -----

// CHECK-LABEL: @extract_finite_only_float(
func.func @extract_finite_only_float(%source: memref<16xf4E2M1FN>, %tile_id: index) -> tensor<8xf4E2M1FN> {
  // CHECK-DAG: %[[PAD:.*]] = arith.constant 0.000000e+00 : f4E2M1FN
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK: scf.if {{.*}} -> (memref<8xf4E2M1FN>) {
  // CHECK: } else {
  // CHECK:   %[[ALLOC_ELSE:.*]] = memref.alloc() : memref<8xf4E2M1FN>
  // CHECK:   scf.for %[[IV:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
  // CHECK:     memref.store %[[PAD]], %[[ALLOC_ELSE]][%[[IV]]] : memref<8xf4E2M1FN>
  // CHECK:   }
  // CHECK: }
  %tile = xtile.extract %source[%tile_id][8][1] : memref<16xf4E2M1FN> -> tensor<8xf4E2M1FN>
  return %tile : tensor<8xf4E2M1FN>
}
