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

func.func @extract_strided_slice() -> vector<2x3xi32> {
  %c = arith.constant dense<[[1,2,3,4],
                             [5,6,7,8],
                             [9,10,11,12]]> : vector<3x4xi32>
  %o = vector.extract_strided_slice %c {
    offsets = [0, 1],
    sizes = [2, 3],
    // TODO(jreiffers): Test non-unit strides when supported by verifier.
    strides = [1, 1]
  } : vector<3x4xi32> to vector<2x3xi32>
  return %o : vector<2x3xi32>
}

// CHECK-LABEL: @extract_strided_slice
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[2, 3, 4], [6, 7, 8]]
