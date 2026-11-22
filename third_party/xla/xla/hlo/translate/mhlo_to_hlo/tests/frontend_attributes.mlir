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
// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s

// Tests parameters with frontend_attributes have such attributes set correctly
// in HloModule

func.func @main(%arg0: tensor<3x4xf32> {mhlo.frontend_attributes = {_test = "a"}}, %arg1: tensor<3x4xf32>) -> tuple<tensor<3x4xf32>, tensor<3x4xf32>> {
  %0 = "mhlo.tuple"(%arg0, %arg1) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tuple<tensor<3x4xf32>, tensor<3x4xf32>>
  func.return %0 : tuple<tensor<3x4xf32>, tensor<3x4xf32>>
}

// CHECK:  ENTRY
// CHECK:  %[[P0:.*]] = f32[3,4] parameter(0), frontend_attributes={_test="a"}
// CHECK:  %[[P1:.*]] = f32[3,4] parameter(1)
// CHECK-NOT: frontend_attributes=

// -----

// Tests call ops with frontend_attributes have such attributes set correctly
// in HloModule

func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = call @add(%arg0, %arg0) {mhlo.frontend_attributes = {_xla_compute_type = "dense"}} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
func.func private @add(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = mhlo.add %arg0, %arg1 : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK:  ENTRY
// CHECK:       %[[CALL:.*]] = f32[] call
// CHECK-SAME:  frontend_attributes={_xla_compute_type="dense"}

// -----

// Tests that the get-tuple-elements unpacking a multi-result call do not
// inherit the call-boundary stream annotation,
// while other frontend attributes are still propagated to them.

func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0:2 = call @two_results(%arg0) {mhlo.frontend_attributes = {_xla_stream_annotation = "locality_domain:0", inlineable = "false", _xla_compute_type = "dense"}} : (tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  func.return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
}
func.func private @two_results(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
  %1 = mhlo.multiply %arg0, %arg0 : tensor<4xf32>
  return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// CHECK:      ENTRY
// CHECK:      %[[CALL:.*]] = (f32[4], f32[4]) call
// CHECK-SAME: frontend_attributes={_xla_compute_type="dense",_xla_stream_annotation="locality_domain:0",inlineable="false"}
// CHECK-DAG:  get-tuple-element(%[[CALL]]), index=0, frontend_attributes={_xla_compute_type="dense",inlineable="false"}
// CHECK-DAG:  get-tuple-element(%[[CALL]]), index=1, frontend_attributes={_xla_compute_type="dense",inlineable="false"}

// -----

// Tests that a regular stream annotation is also kept only on the call.

func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0:2 = call @two_results(%arg0) {mhlo.frontend_attributes = {_xla_stream_annotation = "1", inlineable = "false"}} : (tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  func.return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
}
func.func private @two_results(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
  %1 = mhlo.multiply %arg0, %arg0 : tensor<4xf32>
  return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// CHECK:      ENTRY
// CHECK:      %[[CALL:.*]] = (f32[4], f32[4]) call
// CHECK-SAME: frontend_attributes={_xla_stream_annotation="1",inlineable="false"}
// CHECK-DAG:  get-tuple-element(%[[CALL]]), index=0, frontend_attributes={inlineable="false"}
// CHECK-DAG:  get-tuple-element(%[[CALL]]), index=1, frontend_attributes={inlineable="false"}