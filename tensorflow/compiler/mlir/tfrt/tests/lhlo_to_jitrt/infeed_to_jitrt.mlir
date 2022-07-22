// Copyright 2022 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: lhlo-tfrt-opt %s -lmhlo-to-jitrt | FileCheck %s

// CHECK: @gpu_infeed(
// CHECK:   %[[ARG0:[a-z0-9]+]]: memref<?xf32>
// CHECK: )
func.func @gpu_infeed(%arg0: memref<?xf32>) {
  // CHECK: call @[[INFEED:.*]](%[[ARG0]])
  // CHECK-SAME: {config = "abc"} : (memref<?xf32>) -> ()
  "lmhlo.infeed"(%arg0) {config = "abc"} : (memref<?xf32>) -> ()
  return
}

// CHECK: func private @[[INFEED]](memref<?xf32>)
// CHECK-SAME: attributes {rt.direct_custom_call = "xla.gpu.infeed"}
