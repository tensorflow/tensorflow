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

func.func @tensor() -> tensor<2xi16> {
  %cst = arith.constant dense<[43, 44]> : tensor<2xi16>
  %memref = bufferization.to_buffer %cst : tensor<2xi16> to memref<2xi16>
  %tensor = bufferization.to_tensor %memref : memref<2xi16> to tensor<2xi16>
  return %tensor : tensor<2xi16>
}

// CHECK-LABEL: @tensor
// CHECK{LITERAL}: [43, 44]
