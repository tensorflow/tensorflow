// Copyright 2026 Google Inc. All Rights Reserved.
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
// RUN: tf-opt %s | tf-opt | FileCheck %s

// CHECK-LABEL: func @func_attr
// CHECK-SAME: tf._implements = #tf_type.func<@symbol_a, {attr0 = 1 : i32, attr1 = "random"}>
func.func @func_attr() attributes {tf._implements = #tf_type.func<@symbol_a, {attr0 = 1 : i32, attr1 = "random"}>} {
  func.return
}

// CHECK-LABEL: func @nested_func_attr
// CHECK-SAME: tf._implements = #tf_type.func<@symbol_a, {attr0 = 1 : i32, attr1 = "random", nested = #tf_type.func<@symbol_b, {attr2 = true, attr3 = 8.000000e+00 : f32}>}>
func.func @nested_func_attr() attributes {tf._implements = #tf_type.func<@symbol_a, {attr0 = 1 : i32, attr1 = "random", nested = #tf_type.func<@symbol_b, {attr2 = true, attr3 = 8.0 : f32}>}>} {
  func.return
}
