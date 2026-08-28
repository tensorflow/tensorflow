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
// RUN: xla-translate -mlir-hlo-to-hlo-text -emit-return-tuple %s | FileCheck %s
// RUN: xla-translate -mlir-hlo-to-hlo-text -emit-use-tuple-args -emit-return-tuple %s | FileCheck %s --check-prefix=TUPLE-ARG
// RUN: xla-translate -mlir-hlo-to-hlo-text  %s | FileCheck %s --check-prefix=NO-RETURN-TUPLE

// CHECK-LABEL: ENTRY %main
// CHECK: // OutputIndex {0} aliases with input 0 at {}
// TUPLE-ARG-LABEL: ENTRY %main
// TUPLE-ARG: // OutputIndex {0} aliases with input 0 at {0}
// NO-RETURN-TUPLE-LABEL: ENTRY %main
// NO-RETURN-TUPLE: // OutputIndex {} aliases with input 0 at {}
func.func @main(%arg0: tensor<1xf32> {tf.aliasing_output = 0 : i64}) -> (tensor<1xf32>) {
  %0 = mhlo.constant dense<4.200000e+01> : tensor<1xf32>
  %1 = mhlo.add %arg0, %0 : tensor<1xf32>
  func.return %1 : tensor<1xf32>
}
