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
// RUN: litert-opt %s -split-input-file -tfl-reduce-type-precision -verify-diagnostics

func.func @testI8ToI4WithinRange() -> (tensor<4xi8>) {
  %0 = arith.constant dense<[-8, 0, 1, 7]> : tensor<4xi8>
  // expected-error@+1 {{type of return operand 0 ('tensor<4xi4>') doesn't match function result type ('tensor<4xi8>')}}
  func.return %0 : tensor<4xi8>
}

func.func @testI8ToI4NotWithinRange() -> tensor<4xi8> {
  %0 = arith.constant dense<[-10, 2, 3, 8]> : tensor<4xi8>
  func.return %0 : tensor<4xi8>
}
