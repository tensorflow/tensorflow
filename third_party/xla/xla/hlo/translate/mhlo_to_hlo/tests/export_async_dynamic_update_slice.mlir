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
// RUN: xla-translate --print-sugar=false -split-input-file -mlir-hlo-to-hlo-text -verify-diagnostics %s | FileCheck %s

// The `xla.async_dynamic_update_slice_start` / `_done` custom calls export to
// an HLO async-start / async-done pair wrapping a dynamic-update-slice
// computation, with the async-start output aliased to the operand being
// updated.

// CHECK: HloModule
// CHECK: [[DUS_COMP:%async_dynamic_update_slice.*]] ([[P0:.*]]: f32[4,4], [[P1:.*]]: f32[2,2], [[I0:.*]]: s32[], [[I1:.*]]: s32[]) -> f32[4,4] {
// CHECK:   ROOT {{.*}} = f32[4,4] dynamic-update-slice(%[[P0]], %[[P1]], %[[I0]], %[[I1]])
func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<4x4xf32> {
  %0 = stablehlo.custom_call @xla.async_dynamic_update_slice_start(%arg0, %arg1, %arg2, %arg3) : (tensor<4x4xf32>, tensor<2x2xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
  %1 = stablehlo.custom_call @xla.async_dynamic_update_slice_done(%0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
// CHECK: ENTRY
// CHECK: %[[OPERAND:.*]] = f32[4,4] parameter(0)
// CHECK: %[[UPDATE:.*]] = f32[2,2] parameter(1)
// CHECK: %[[IDX0:.*]] = s32[] parameter(2)
// CHECK: %[[IDX1:.*]] = s32[] parameter(3)
// CHECK: %[[START:.*]] = ((f32[4,4], f32[2,2], s32[], s32[]), f32[4,4], s32[]) async-start(%[[OPERAND]], %[[UPDATE]], %[[IDX0]], %[[IDX1]])
// CHECK-SAME{LITERAL}: output_to_operand_aliasing={{1}: (0, {})}
// CHECK-SAME: calls=[[DUS_COMP]]
// CHECK: ROOT {{.*}} = f32[4,4] async-done(%[[START]])

// -----

// The MHLO custom call spelling exports the same way.

// CHECK: HloModule
// CHECK: [[DUS_COMP:%async_dynamic_update_slice.*]] ({{.*}}: f32[4,4], {{.*}}: f32[2,2], {{.*}}: s32[], {{.*}}: s32[]) -> f32[4,4] {
// CHECK:   ROOT {{.*}} = f32[4,4] dynamic-update-slice(
func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<4x4xf32> {
  %0 = mhlo.custom_call @xla.async_dynamic_update_slice_start(%arg0, %arg1, %arg2, %arg3) : (tensor<4x4xf32>, tensor<2x2xf32>, tensor<i32>, tensor<i32>) -> tensor<4x4xf32>
  %1 = mhlo.custom_call @xla.async_dynamic_update_slice_done(%0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
// CHECK: ENTRY
// CHECK: %[[START:.*]] = ((f32[4,4], f32[2,2], s32[], s32[]), f32[4,4], s32[]) async-start(
// CHECK-SAME{LITERAL}: output_to_operand_aliasing={{1}: (0, {})}
// CHECK-SAME: calls=[[DUS_COMP]]
// CHECK: ROOT {{.*}} = f32[4,4] async-done(%[[START]])

// -----

// The async bundle may flow through an optimization barrier (with interleaved
// compute) before it reaches the done.

// CHECK: HloModule
func.func @main(%arg0: tensor<8x8x128xf32>, %arg1: tensor<2x8x128xf32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>) -> (tensor<8x8x128xf32>, tensor<8x8x128xf32>) {
  %0 = stablehlo.custom_call @xla.async_dynamic_update_slice_start(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<8x8x128xf32>, tensor<2x8x128xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<8x8x128xf32>
  %1 = stablehlo.add %arg0, %arg0 : tensor<8x8x128xf32>
  %2:2 = stablehlo.optimization_barrier %0, %1 : tensor<8x8x128xf32>, tensor<8x8x128xf32>
  %3 = stablehlo.custom_call @xla.async_dynamic_update_slice_done(%2#0) : (tensor<8x8x128xf32>) -> tensor<8x8x128xf32>
  return %3, %2#1 : tensor<8x8x128xf32>, tensor<8x8x128xf32>
}
// CHECK: ENTRY
// CHECK: %[[START:.*]] = ((f32[8,8,128], f32[2,8,128], s32[], s32[], s32[]), f32[8,8,128], s32[]) async-start(
// CHECK-SAME{LITERAL}: output_to_operand_aliasing={{1}: (0, {})}
// CHECK: %[[ADD:.*]] = f32[8,8,128] add(
// CHECK: %[[TUPLE:.*]] = (((f32[8,8,128], f32[2,8,128], s32[], s32[], s32[]), f32[8,8,128], s32[]), f32[8,8,128]) tuple(%[[START]], %[[ADD]])
// CHECK: %[[BARRIER:.*]] = (((f32[8,8,128], f32[2,8,128], s32[], s32[], s32[]), f32[8,8,128], s32[]), f32[8,8,128]) opt-barrier(%[[TUPLE]])
// CHECK: %[[BUNDLE:.*]] = ((f32[8,8,128], f32[2,8,128], s32[], s32[], s32[]), f32[8,8,128], s32[]) get-tuple-element(%[[BARRIER]]), index=0
// CHECK: %[[DONE:.*]] = f32[8,8,128] async-done(%[[BUNDLE]])
// CHECK: %[[OTHER:.*]] = f32[8,8,128] get-tuple-element(%[[BARRIER]]), index=1
// CHECK: ROOT {{.*}} = (f32[8,8,128], f32[8,8,128]) tuple(%[[DONE]], %[[OTHER]])

// -----

// Two in flight updates with different ranks and element types export to
// distinct async-start / async-done pairs, each with its own computation.

// CHECK: HloModule
// CHECK: [[DUS_1D:%async_dynamic_update_slice.*]] ({{.*}}: bf16[16], {{.*}}: bf16[4], {{.*}}: s32[]) -> bf16[16] {
// CHECK:   ROOT {{.*}} = bf16[16] dynamic-update-slice(
// CHECK: [[DUS_3D:%async_dynamic_update_slice.*]] ({{.*}}: s32[4,8,16], {{.*}}: s32[1,8,16], {{.*}}: s32[], {{.*}}: s32[], {{.*}}: s32[]) -> s32[4,8,16] {
// CHECK:   ROOT {{.*}} = s32[4,8,16] dynamic-update-slice(
func.func @main(%x1: tensor<16xbf16>, %u1: tensor<4xbf16>, %i0: tensor<i32>, %x2: tensor<4x8x16xi32>, %u2: tensor<1x8x16xi32>, %j0: tensor<i32>, %j1: tensor<i32>, %j2: tensor<i32>) -> (tensor<16xbf16>, tensor<4x8x16xi32>) {
  %f1 = stablehlo.custom_call @xla.async_dynamic_update_slice_start(%x1, %u1, %i0) : (tensor<16xbf16>, tensor<4xbf16>, tensor<i32>) -> tensor<16xbf16>
  %f2 = stablehlo.custom_call @xla.async_dynamic_update_slice_start(%x2, %u2, %j0, %j1, %j2) : (tensor<4x8x16xi32>, tensor<1x8x16xi32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4x8x16xi32>
  %r2 = stablehlo.custom_call @xla.async_dynamic_update_slice_done(%f2) : (tensor<4x8x16xi32>) -> tensor<4x8x16xi32>
  %r1 = stablehlo.custom_call @xla.async_dynamic_update_slice_done(%f1) : (tensor<16xbf16>) -> tensor<16xbf16>
  return %r1, %r2 : tensor<16xbf16>, tensor<4x8x16xi32>
}
// CHECK: ENTRY
// CHECK: %[[START_1D:.*]] = ((bf16[16], bf16[4], s32[]), bf16[16], s32[]) async-start(
// CHECK-SAME{LITERAL}: output_to_operand_aliasing={{1}: (0, {})}
// CHECK-SAME: calls=[[DUS_1D]]
// CHECK: %[[DONE_1D:.*]] = bf16[16] async-done(%[[START_1D]])
// CHECK: %[[START_3D:.*]] = ((s32[4,8,16], s32[1,8,16], s32[], s32[], s32[]), s32[4,8,16], s32[]) async-start(
// CHECK-SAME{LITERAL}: output_to_operand_aliasing={{1}: (0, {})}
// CHECK-SAME: calls=[[DUS_3D]]
// CHECK: %[[DONE_3D:.*]] = s32[4,8,16] async-done(%[[START_3D]])
// CHECK: ROOT {{.*}} = (bf16[16], s32[4,8,16]) tuple(%[[DONE_1D]], %[[DONE_3D]])

// -----

// expected-error@-3 {{xla.async_dynamic_update_slice_start requires at least 3 operands}}
func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<2x2xf32>) -> tensor<4x4xf32> {
  %0 = stablehlo.custom_call @xla.async_dynamic_update_slice_start(%arg0, %arg1) : (tensor<4x4xf32>, tensor<2x2xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// -----

// expected-error@-3 {{xla.async_dynamic_update_slice_done requires at least 1 operand}}
func.func @main() -> tensor<4x4xf32> {
  %0 = stablehlo.custom_call @xla.async_dynamic_update_slice_done() : () -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
