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
// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @create_mask() -> vector<4x3xi1> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %1 = vector.create_mask %c3, %c2 : vector<4x3xi1>
  return %1 : vector<4x3xi1>
}

// CHECK-LABEL: @create_mask
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<4x3xi1>:
// CHECK-SAME{LITERAL}: [[true, true, false],
// CHECK-SAME{LITERAL}:  [true, true, false],
// CHECK-SAME{LITERAL}:  [true, true, false],
// CHECK-SAME{LITERAL}:  [false, false, false]]
