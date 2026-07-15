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
// RUN: xla-opt %s -split-input-file \
// RUN: -extract-tma-info \
// RUN: | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
tt.func @extract_tma_info_128b(%arg0: !tt.tensordesc<16x32xf32, #shared>
  {tt.nv_tma_desc = 1 : i32,
   tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [32, 256],
                                                  tile_shape = [16, 32],
                                                  tile_strides = [1, 1],
                                                  layout = [1, 0],
                                                  element_byte_size = 4>}) {
  tt.return
}

// CHECK-LABEL: tt.func @extract_tma_info_128b
// CHECK-SAME:  %[[ARG_0:.*]]: !tt.tensordesc<16x32xf32, #shared>
// CHECK-SAME: #triton_xla.tma_descriptor<global_shape = [32, 256],
// CHECK-SAME: tile_shape = [16, 32], tile_strides = [1, 1], layout = [1, 0],
// CHECK-SAME: element_byte_size = 4, swizzle_mode = "128b">

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 32}>
tt.func @extract_tma_info_64b(%arg0: !tt.tensordesc<16x32xf32, #shared>
  {tt.nv_tma_desc = 1 : i32,
   tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [32, 256],
                                                  tile_shape = [16, 32],
                                                  tile_strides = [1, 1],
                                                  layout = [1, 0],
                                                  element_byte_size = 4>}) {
  tt.return
}

// CHECK-LABEL: tt.func @extract_tma_info_64b
// CHECK-SAME:  %[[ARG_0:.*]]: !tt.tensordesc<16x32xf32, #shared>
// CHECK-SAME: #triton_xla.tma_descriptor<global_shape = [32, 256],
// CHECK-SAME: tile_shape = [16, 32], tile_strides = [1, 1], layout = [1, 0],
// CHECK-SAME: element_byte_size = 4, swizzle_mode = "64b">

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 32}>
tt.func @extract_tma_info_32b(%arg0: !tt.tensordesc<16x32xf32, #shared>
  {tt.nv_tma_desc = 1 : i32,
   tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [32, 256],
                                                  tile_shape = [16, 32],
                                                  tile_strides = [1, 1],
                                                  layout = [1, 0],
                                                  element_byte_size = 4>}) {
  tt.return
}

// CHECK-LABEL: tt.func @extract_tma_info_32b
// CHECK-SAME:  %[[ARG_0:.*]]: !tt.tensordesc<16x32xf32, #shared>
// CHECK-SAME: #triton_xla.tma_descriptor<global_shape = [32, 256],
// CHECK-SAME: tile_shape = [16, 32], tile_strides = [1, 1], layout = [1, 0],
// CHECK-SAME: element_byte_size = 4, swizzle_mode = "32b">

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
tt.func @extract_tma_info_swizzled_shared(%arg0: !tt.tensordesc<16x32xf32, #shared>
  {tt.nv_tma_desc = 1 : i32,
   tt.tma_descriptor = #triton_xla.tma_descriptor<global_shape = [32, 256],
                                                  tile_shape = [16, 32],
                                                  tile_strides = [1, 1],
                                                  layout = [1, 0],
                                                  element_byte_size = 4>}) {
  tt.return
}
// CHECK-LABEL: tt.func @extract_tma_info_swizzled_shared
// CHECK-SAME:  %[[ARG_0:.*]]: !tt.tensordesc<16x32xf32, #shared>
// CHECK-SAME: #triton_xla.tma_descriptor<global_shape = [32, 256],
// CHECK-SAME: tile_shape = [16, 32], tile_strides = [1, 1], layout = [1, 0],
// CHECK-SAME: element_byte_size = 4, swizzle_mode = "none">
