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
// RUN: emitters_opt %s --split-input-file -verify-roundtrip

func.func @load(%arg0: !xla_cpu.call_frame) -> tensor<32x32xf32> {
  %0 = xla_cpu.load %arg0, 0 : tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// -----

func.func @load(%arg0: !xla_cpu.call_frame) -> memref<64x32xf32> {
  %0 = xla_cpu.load %arg0, 0 : memref<64x32xf32>
  return %0 : memref<64x32xf32>
}

// -----

func.func @extract_workgroup_id(%arg0: !xla_cpu.call_frame) -> index {
  %0 = xla_cpu.extract_workgroup_id %arg0, x
  return %0 : index
}
