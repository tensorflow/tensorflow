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

// Exports the TFLite element types (`!tfl.string`, `!tfl.resource`).

// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

// CHECK-DAG: type: STRING,
// CHECK-DAG: type: RESOURCE,
// CHECK: builtin_options_type: HashtableOptions,
// CHECK-NEXT: builtin_options: {
// CHECK-NEXT:   table_id: 1,
// CHECK-NEXT:   key_dtype: STRING,
// CHECK-NEXT:   value_dtype: INT64
func.func @main(%arg0: tensor<2x!tfl.string>) -> tensor<2xi64> {
  %0 = "tfl.hashtable"() {table_id = 1 : i32, key_dtype = !tfl.string, value_dtype = i64} : () -> tensor<1x!tfl.resource>
  %cst = "tfl.pseudo_const"() {value = dense<-1> : tensor<i64>} : () -> tensor<i64>
  %1 = "tfl.hashtable_find"(%0, %arg0, %cst) : (tensor<1x!tfl.resource>, tensor<2x!tfl.string>, tensor<i64>) -> tensor<2xi64>
  func.return %1 : tensor<2xi64>
}
