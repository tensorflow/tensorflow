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

func.func @bitcast_convert() -> tensor<3xf32> {
  %c = arith.constant dense<[10000000,20000000,30000000]> : tensor<3xi32>
  %ret = mhlo.bitcast_convert %c : (tensor<3xi32>) -> tensor<3xf32>
  return %ret : tensor<3xf32>
}

// CHECK-LABEL: @bitcast_convert
// CHECK-NEXT: Results
// CHECK-NEXT: [1.401298e-38, 3.254205e-38, 7.411627e-38]
