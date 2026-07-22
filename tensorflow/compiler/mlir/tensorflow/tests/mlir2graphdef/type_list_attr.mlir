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
    // CHECK:       key: "emptylist"
    // CHECK-NEXT:  value {
    // CHECK-NEXT:    list {
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    // CHECK:       key: "typelist"
    // CHECK-NEXT:  value {
    // CHECK-NEXT:    list {
    // CHECK-NEXT:      type: DT_INT32
    // CHECK-NEXT:      type: DT_FLOAT
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    %0:2 = tf_executor.island wraps "tf.Placeholder"() {name = "dummy", dtype = "tfdtype$DT_FLOAT", emptylist = [], typelist = ["tfdtype$DT_INT32", "tfdtype$DT_FLOAT"]} : () -> tensor<*xi32>
    tf_executor.fetch
  }
  func.return
}
