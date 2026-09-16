// Copyright 2026 Google Inc. All Rights Reserved.
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
// RUN: tf-opt %s -tf-strip-noinline-attribute | FileCheck %s

// CHECK-LABEL: func @strip_simple(
// CHECK-NOT: tf._noinline
func.func @strip_simple() -> tensor<2xi32> attributes {tf._noinline = true} {
  // CHECK-NEXT: %[[CST:.*]] = "tf.Const"
  %cst = "tf.Const"() { value = dense<2> : tensor<2xi32> } : () -> tensor<2xi32>
  func.return %cst : tensor<2xi32>
}
