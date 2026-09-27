// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
// RUN: tf-opt "-tfxla-verify-legalization=legalize-chlo=false" -verify-diagnostics -split-input-file %s | FileCheck %s --dump-input=fail
// Tests the VerifyTFXLALegalization Pass, that just ensures we don't have
// any illegal ops at the end of the pipeline. This runs with
// legalize-chlo=false since errors can't be mixed with the legalize-chlo=True
// version.

// CHECK-LABEL: allows_chlo
func.func @allows_chlo(%arg0: tensor<1x32x10x32xi32>, %arg1: tensor<32xi32>) -> tensor<1x32x10x32xi32> {
  %0 = "chlo.broadcast_add"(%arg0, %arg1) {broadcast_dimensions = array<i64: 3>} : (tensor<1x32x10x32xi32>, tensor<32xi32>) -> tensor<1x32x10x32xi32>
  func.return %0 : tensor<1x32x10x32xi32>
}
