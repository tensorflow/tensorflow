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
// RUN: tf-mlir-translate -mlir-to-graphdef %s -tf-export-original-func-name | tf-mlir-translate -graphdef-to-mlir | tf-mlir-translate -mlir-to-graphdef -tf-export-original-func-name | FileCheck %s

// Tests #tf_type.func attributes are exported as AttrValue.NameAttrList attributes
// with its attr field populated with nested attributes.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 458 : i32}} {
  func.func @main() {
    tf_executor.graph {
      %control = tf_executor.island wraps "tf.NoOp"() {_f = #tf_type.func<@callee, {attr2 = true, attr3 = 8.0 : f32}>} : () -> ()
      %control_1 = tf_executor.island(%control) wraps "tf.LegacyCall"() {f = @callee} : () -> ()
      tf_executor.fetch
    }
    func.return
  }
  func.func @callee() attributes {tf._original_func_name = "original_callee"} {
    tf_executor.graph {
      tf_executor.fetch
    }
    func.return
  }
}

// CHECK:        op: "NoOp"
// CHECK-NEXT:   attr
// CHECK-NEXT:     key: "_f"
// CHECK-NEXT:     value
// CHECK-NEXT:       func
// CHECK-NEXT:         name: "original_callee"
// CHECK-NEXT:         attr
// CHECK-NEXT:           key: "attr2"
// CHECK-NEXT:           value
// CHECK-NEXT:             b: true
// CHECK:              attr
// CHECK-NEXT:           key: "attr3"
// CHECK-NEXT:           value
// CHECK-NEXT:             f: 8

// CHECK:        op: "original_callee"

// CHECK:      library
// CHECK-NEXT:   function
// CHECK-NEXT:     signature
// CHECK-NEXT:       name: "original_callee"
