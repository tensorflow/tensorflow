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
// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func.func @main() {
  tf_executor.graph {
    tf_executor.island wraps "tf.NoOp"() {} : () -> () loc("noop")
    tf_executor.fetch
  }
  func.return
}

// CHECK: node {
// CHECK-NEXT:  name: "noop"
// CHECK-NEXT:  op: "NoOp"
// CHECK-NEXT:  experimental_debug_info {
// CHECK-NEXT:  }
// CHECK-NEXT: }
