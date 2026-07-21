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
// RUN: litert-opt %s -tfl-legalize-tf='preserve-assert-op=true' | FileCheck %s --dump-input=fail

func.func @preserve_assert(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<1xi1> {
  %0 = "tf.LessEqual"(%arg0, %arg1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
  "tf.Assert"(%0, %arg1) {summarize = 3} : (tensor<1xi1>, tensor<1xi32>) -> ()
  func.return %0 : tensor<1xi1>
  // CHECK-LABEL: preserve_assert
  // CHECK: tfl.less_equal
  // CHECK: Assert
  // CHECK: return
}
