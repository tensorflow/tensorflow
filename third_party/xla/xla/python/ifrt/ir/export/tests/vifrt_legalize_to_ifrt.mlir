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
// RUN: ifrt-opt %s --vifrt-legalize-to-ifrt | FileCheck %s

// Verifies that legacy VIFRT arrays carrying literal `"device"` / `"default"`
// strings canonicalize to default IFRT attributes (omitted when printed) while
// non-default attributes are preserved.
!v_explicit_def = !vifrt.array_v1<tensor<2x4xi32>,
                                  #vifrt.sharding_param_v2<1x1 to [0] on 2>,
                                  [0, 1], memory_kind = "device",
                                  layout = "default">
!v_non_def = !vifrt.array_v1<tensor<2x4xi32>,
                             #vifrt.sharding_param_v2<1x1 to [0] on 2>, [2, 3],
                             memory_kind = "pinned_host", layout = "{0,1}">
// CHECK: #sp = #ifrt.sharding_param<1x1 to [0] on 2>
// CHECK-LABEL: func.func @legacy_vifrt_arrays
// CHECK-SAME: (%[[ARG0:.+]]: !ifrt.array<tensor<2x4xi32>, #sp, [0, 1]>,
// CHECK-SAME:  %[[ARG1:.+]]: !ifrt.array<tensor<2x4xi32>, #sp, [2, 3], memory_kind = "pinned_host", layout = "{0,1}">)
// CHECK-SAME: -> (!ifrt.array<tensor<2x4xi32>, #sp, [0, 1]>, !ifrt.array<tensor<2x4xi32>, #sp, [2, 3], memory_kind = "pinned_host", layout = "{0,1}">)
vifrt.FuncV1 @legacy_vifrt_arrays(%arg0: !v_explicit_def, %arg1: !v_non_def)
    -> (!v_explicit_def, !v_non_def) {
  // CHECK: return %[[ARG0]], %[[ARG1]]
  "vifrt.ReturnV1"(%arg0, %arg1) : (!v_explicit_def, !v_non_def) -> ()
} <{arg_attrs = [], res_attrs = [], sym_visibility = "vifrt.default"}> {vifrt.function}
