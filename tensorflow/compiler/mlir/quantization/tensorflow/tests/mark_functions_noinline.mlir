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
// RUN: tf-quant-opt %s -mark-functions-noinline='noinline-functions=noinline0' \
// RUN:     -allow-unregistered-dialect -mlir-disable-threading \
// RUN:     -split-input-file -verify-diagnostics | FileCheck %s

// Tests that the function is marked tf._noinline = true.

// CHECK-LABEL: @noinline0
// CHECK-SAME: attributes {{{.*tf._noinline = true.*}}}
func.func @noinline0() -> (tensor<0xf32>) {
  %cst = "tf.Const"() {value = dense<1.0> : tensor<0xf32>} : () -> tensor<0xf32>
  return %cst : tensor<0xf32>
}

// -----

// Tests that the function not listed in the option `noinline-functions`
// is not marked tf._noinline = true.

// CHECK-LABEL: @inline
// CHECK-NOT: tf._noinline
func.func @inline() -> (tensor<0xf32>) {
  %cst = "tf.Const"() {value = dense<1.0> : tensor<0xf32>} : () -> tensor<0xf32>
  return %cst : tensor<0xf32>
}
