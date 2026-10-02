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
// RUN: litert-opt -tfl-trim-funcs-tf="trim-funcs-allowlist=bar,foobar" %s | FileCheck %s

func.func @foo(%arg0: tensor<1x4xf32>, %arg1: tensor<1x4xf32>) -> tensor<1x4xf32> {
  func.return %arg0 : tensor<1x4xf32>
}

func.func @bar(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
  func.return %arg0 : tensor<2x4xf32>
}

func.func @foobar(%arg0: tensor<1x4xf32>, %arg1: tensor<1x4xf32>) -> tensor<1x4xf32> {
  func.return %arg0 : tensor<1x4xf32>
}

// CHECK-DAG: func @main
// CHECK-DAG: func @foobar
// CHECK-NOT: func @foo
// CHECK-NOT: func @bar
