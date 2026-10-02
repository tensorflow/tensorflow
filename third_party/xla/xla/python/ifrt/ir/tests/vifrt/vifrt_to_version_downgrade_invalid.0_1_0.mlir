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
// RUN: ifrt-opt %s --ifrt-legalize-to-vifrt --vifrt-to-version='target_version=0.1.0' --symbol-dce --verify-diagnostics --split-input-file

!array = !ifrt.array<tensor<6xi32>,
                     #ifrt.sharding_param<2 to [0,1] on 2x2 unreduced [1]>,
                     [0, 1, 2, 3]>

// expected-error@-21 {{failed to convert to VIFRT version 0.1.0}}
// expected-error@+1 {{failed to legalize operation 'vifrt.FuncV1' that was explicitly marked illegal}}
func.func @array_with_unreduced_axes(%arg0: !array) attributes {ifrt.function} {
  %0, %ctrl = ifrt.CopyArrays(%arg0) : (!array) -> !array
  return
}
