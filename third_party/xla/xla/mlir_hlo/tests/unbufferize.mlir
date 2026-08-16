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
// RUN: mlir-hlo-opt --unbufferize %s | FileCheck %s

// CHECK-LABEL: func @unbufferize
// CHECK-SAME: (%arg0: tensor<8xf32>) -> (tensor<8xf32> {my.attr})
func.func @unbufferize(%arg0: memref<8xf32>, %arg1: memref<8xf32> {my.attr}) {
  %0 = bufferization.to_tensor %arg0 : memref<8xf32> to tensor<8xf32>
  bufferization.materialize_in_destination %0 in writable %arg1
      : (tensor<8xf32>, memref<8xf32>) -> ()
  // CHECK-NEXT: return %arg0 : tensor<8xf32>
  return
}

// CHECK-LABEL: func @not_block_arg
func.func @not_block_arg() {
  %0 = memref.alloc() : memref<8xf32>
  // CHECK: bufferization.to_tensor
  %1 = bufferization.to_tensor %0 : memref<8xf32> to tensor<8xf32>
  // CHECK: bufferization.materialize_in_destination
  bufferization.materialize_in_destination %1 in writable %0
      : (tensor<8xf32>, memref<8xf32>) -> ()
  return
}
