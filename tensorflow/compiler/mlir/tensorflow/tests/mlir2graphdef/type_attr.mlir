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

// Check that attributes that define types are exported.

// CHECK: key: "Tinputs"
// CHECK-NEXT:    value
// CHECK-NEXT:      list
// CHECK-NEXT:        type: DT_FLOAT

// CHECK: key: "Toutputs"
// CHECK-NEXT:    value
// CHECK-NEXT:      list
// CHECK-NEXT:        type: DT_FLOAT

// CHECK: "extra_type_attr"
// CHECK-NEXT:    value
// CHECK-NEXT:      list
// CHECK-NEXT:        type: DT_INT32
// CHECK-NEXT:        type: DT_FLOAT

// CHECK-LABEL: function
// CHECK: name: "plain"
// CHECK: Placeholder
// CHECK: key: "type"
// CHECK: type: DT_INT8

func.func @main(%arg0 : tensor<16xf32>) {
  tf_executor.graph {
    %1:2 = tf_executor.island wraps "tf.MlirPassthroughOp"(%arg0) {extra_type_attr = [tensor<5xi32>, tensor<16xf32>], Tinputs = [tensor<16xf32>], Toutputs = [tensor<16xf32>], mlir_module = ""} : (tensor<16xf32>) -> tensor<16xf32>
    tf_executor.fetch
  }
  func.return
}

func.func @plain() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Placeholder"() {type = i8} : () -> tensor<16xi8>
    tf_executor.fetch
  }
  func.return
}
