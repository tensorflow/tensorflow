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
// RUN: xla-opt %s --split-input-file | FileCheck %s

tt.func @tma_descriptor_params(%arg0: tensor<512x128xf32>)
  -> tensor<512x128xf32> attributes {
    tma = #triton_xla.tma_descriptor<
      global_shape = [512, 128],
      tile_shape = [32, 64],
      tile_strides = [1, 1],
      layout = [0, 1],
      element_byte_size = 4>
  } {
  tt.return %arg0  : tensor<512x128xf32>
}
// CHECK:  #tma_descriptor =  #triton_xla.tma_descriptor<
// CHECK-SAME:   global_shape = [512, 128],
// CHECK-SAME:   tile_shape = [32, 64],
// CHECK-SAME:   tile_strides = [1, 1],
// CHECK-SAME:   layout = [0, 1],
// CHECK-SAME:   element_byte_size = 4>

// -----

tt.func @tma_descriptor_params(%arg0: tensor<512x128xf32>)
  -> tensor<512x128xf32> attributes {
    tma = #triton_xla.tma_descriptor<
      global_shape = [512, 128],
      tile_shape = [32, 64],
      tile_strides = [1, 1],
      layout = [0, 1],
      element_byte_size = 4,
      swizzle_mode = "32b">
  } {
  tt.return %arg0  : tensor<512x128xf32>
}
// CHECK:  #tma_descriptor =  #triton_xla.tma_descriptor<
// CHECK-SAME:   global_shape = [512, 128],
// CHECK-SAME:   tile_shape = [32, 64],
// CHECK-SAME:   tile_strides = [1, 1],
// CHECK-SAME:   layout = [0, 1],
// CHECK-SAME:   element_byte_size = 4,
// CHECK-SAME:   swizzle_mode = "32b">