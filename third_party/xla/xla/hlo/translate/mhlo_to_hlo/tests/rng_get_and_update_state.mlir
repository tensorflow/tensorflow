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
// RUN: xla-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

// CHECK-LABEL: ENTRY %main.{{.*}} () -> u64[2]
// CHECK-NEXT: ROOT %rng-get-and-update-state.1 = u64[2] rng-get-and-update-state(), delta=1
func.func @main() -> tensor<2xui64> {
  %1 = mhlo.xla.rng_get_and_update_state {delta = 1: i64}
  func.return %1 : tensor<2xui64>
}
