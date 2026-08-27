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
// RUN: kernel-gen-opt %s -split-input-file -verify-diagnostics

func.func @alloc_raw(%ctx: !tf_framework.op_kernel_context, %size : index) {
  // expected-error @+1 {{`dyn_sizes` count 1 does not match dynamic dimensions}}
  %buf = tf_framework.alloc(%ctx, %size) : memref<?x10x?xi8>
  func.return
}
