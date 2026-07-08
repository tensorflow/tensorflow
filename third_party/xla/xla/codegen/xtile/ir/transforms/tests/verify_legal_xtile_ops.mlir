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
// RUN: emitters_opt %s -xtile-verify-legal-ops -split-input-file -verify-diagnostics

xtile.entry_func @fails_illegal_op(%arg0: memref<2xf32>, %arg1: index) {
  %c_0 = arith.constant 0. : f32
  // expected-error @+1 {{vector.transfer_read: unsupported op}}
  %0 = vector.transfer_read %arg0[%arg1], %c_0 : memref<2xf32>, vector<2xf32>
  // expected-error @+1 {{vector.transfer_write: unsupported op}}
  vector.transfer_write %0, %arg0[%arg1] : vector<2xf32>, memref<2xf32>
  xtile.return
}

// -----

func.func @iota_2d_fails() -> tensor<2x2xi32> {
  // expected-error @+1 {{Only 1D iota is supported}}
  %0 = stablehlo.iota dim = 0 :  tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}
