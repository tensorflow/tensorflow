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

// RUN: emitters_opt %s --allow-unregistered-dialect --verify-diagnostics \
// RUN: -xla-lower-tensors="gpu_device_info='cuda_compute_capability {major: 6}'" \
// RUN: | FileCheck %s

// A tensor.insert into the result of xla_gpu.sync_threads must resolve to the
// shared allocation without diagnostics.
func.func @insert_after_sync_threads(%v: f32, %w: f32, %i: index,
                                     %j: index) -> tensor<32xf32> {
  %shared = xla_gpu.allocate_shared : tensor<32xf32>
  %written = tensor.insert %v into %shared[%i] : tensor<32xf32>
  %synced = xla_gpu.sync_threads %written : tensor<32xf32>
  %r = tensor.insert %w into %synced[%j] : tensor<32xf32>
  return %r : tensor<32xf32>
}
// CHECK-LABEL: @insert_after_sync_threads
// CHECK:       %[[ADDR:.*]] = llvm.mlir.addressof
// CHECK:       %[[CAST:.*]] = llvm.addrspacecast %[[ADDR]]
// CHECK:       %[[PTR0:.*]] = llvm.getelementptr inbounds %[[CAST]]
// CHECK:       llvm.store %{{.*}}, %[[PTR0]] : f32, !llvm.ptr
// CHECK:       gpu.barrier
// CHECK:       %[[PTR1:.*]] = llvm.getelementptr inbounds %[[CAST]]
// CHECK:       llvm.store %{{.*}}, %[[PTR1]] : f32, !llvm.ptr
