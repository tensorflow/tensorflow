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
// RUN: emitters_opt %s -split-input-file -xla-cpu-add-reduction-fast-math-flags | FileCheck %s

func.func @caller(%x: f32, %y: f32) -> f32
{
  %z = func.call @reducer(%x, %y) { xla.is_reduction }: (f32, f32) -> f32
  func.return %z : f32
}

func.func @reducer(%x: f32, %y: f32) -> f32
{
  %z = arith.addf %x, %y : f32
  func.return %z : f32
}

// CHECK-LABEL: func.func @caller
// CHECK-LABEL: func.func @reducer
// CHECK: arith.addf {{.*}} fastmath<reassoc> : f32

// -----


func.func @caller(%x: f32, %y: f32) -> f32
{
  %z = func.call @reducer(%x, %y) { xla.is_reduction }: (f32, f32) -> f32
  func.return %z : f32
}

func.func @reducer(%x: f32, %y: f32) -> f32
{
  %w = arith.addf %x, %y : f32
  %z = arith.mulf %w, %y : f32
  func.return %z : f32
}

// CHECK-LABEL: func.func @caller
// CHECK-LABEL: func.func @reducer
// CHECK-NOT: fastmath
