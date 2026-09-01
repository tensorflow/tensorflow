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
// RUN: tf-opt %s --test-tf-lower-tf --xla-legalize-tf | \
// RUN: mlir-hlo-opt \
// RUN: --hlo-legalize-to-stablehlo=allow-xla-features \
// RUN: --stablehlo-legalize-to-linalg \
// RUN: --empty-tensor-to-alloc-tensor \
// RUN: --computeop-and-func-bufferize --canonicalize | \
// RUN: kernel-gen-opt -allow-unregistered-dialect \
// RUN: --shape-to-descriptors \
// RUN: --canonicalize --kernelgen-final-bufferize | \
// RUN: FileCheck %s

// Test whether all shape computations required for isinf can be lowered to
// the standard dialect, scf and descriptors.
// CHECK-LABEL: @isinf
func.func @isinf(%arg0: tensor<?xf32>) -> tensor<?xi1> {
  // CHECK-NOT: shape
  %0 = "tf.IsInf"(%arg0) : (tensor<?xf32>) -> tensor<?xi1>
  func.return %0 : tensor<?xi1>
}
